"""
Persistent Cross-Meeting Memory & Organizational Intelligence Engine.
Stores, indexes, and recalls past meeting decisions, ongoing workstreams, and known participants.
"""

import asyncio
from datetime import datetime, timezone
import json
import logging
from typing import Any

from app.ai_brain.models import (
    ActionItem,
    Decision,
    M2ToM3Contract,
    MeetingIntelligenceResult,
    MemoryRecord,
    MemoryStore,
)
from app.infrastructure import SQLiteDatabase

logger = logging.getLogger("automation.ai_brain.memory")


class InMemoryMemoryStore:
    """In-memory local memory adapter for tests and local sessions."""

    def __init__(self) -> None:
        self._records: list[MemoryRecord] = []
        self._lock = asyncio.Lock()

    async def recall(
        self,
        client_id: str | None,
        limit: int,
    ) -> list[MemoryRecord]:
        matching = [
            record for record in self._records if client_id is None or record.client_id == client_id
        ]
        return sorted(matching, key=lambda record: record.occurred_at, reverse=True)[:limit]

    async def save(self, record: MemoryRecord) -> None:
        async with self._lock:
            self._records = [
                existing for existing in self._records if existing.meeting_id != record.meeting_id
            ]
            self._records.append(record)


class SQLiteMemoryStore:
    """Persistent SQLite memory store that persists and recalls cross-meeting intelligence across runs."""

    def __init__(self, db: SQLiteDatabase | None = None) -> None:
        self.db = db or SQLiteDatabase()
        self._lock = asyncio.Lock()
        self._ensure_extended_schema()

    def _ensure_extended_schema(self) -> None:
        """Adds new columns for title, decisions, participants if upgrading an existing db."""
        try:
            with self.db.connect() as conn:
                cursor = conn.execute("PRAGMA table_info(meeting_memory);")
                existing_cols = {row["name"] for row in cursor.fetchall()}
                if "title" not in existing_cols:
                    conn.execute("ALTER TABLE meeting_memory ADD COLUMN title TEXT DEFAULT '';")
                if "decisions_json" not in existing_cols:
                    conn.execute("ALTER TABLE meeting_memory ADD COLUMN decisions_json TEXT DEFAULT '[]';")
                if "participants_json" not in existing_cols:
                    conn.execute("ALTER TABLE meeting_memory ADD COLUMN participants_json TEXT DEFAULT '[]';")
        except Exception as ex:
            logger.warning("Could not run extended schema migration: %s", ex)

    async def recall(
        self,
        client_id: str | None,
        limit: int,
    ) -> list[MemoryRecord]:
        try:
            if client_id is not None:
                rows = await self.db.execute_query(
                    """
                    SELECT meeting_id, title, client_id, occurred_at, summary,
                           decisions_json, pending_actions_json, participants_json
                    FROM meeting_memory WHERE client_id = ?
                    ORDER BY occurred_at DESC LIMIT ?
                    """,
                    (client_id, limit),
                )
            else:
                rows = await self.db.execute_query(
                    """
                    SELECT meeting_id, title, client_id, occurred_at, summary,
                           decisions_json, pending_actions_json, participants_json
                    FROM meeting_memory
                    ORDER BY occurred_at DESC LIMIT ?
                    """,
                    (limit,),
                )
        except Exception:
            # Fallback for old schema if columns not yet loaded
            rows = await self.db.execute_query(
                "SELECT meeting_id, client_id, occurred_at, summary, pending_actions_json FROM meeting_memory ORDER BY occurred_at DESC LIMIT ?",
                (limit,),
            )

        records = []
        for row in rows:
            try:
                row_dict = dict(row)
                actions_raw = json.loads(row_dict.get("pending_actions_json") or "[]")
                actions = [ActionItem.model_validate(act) for act in actions_raw]
                
                decisions_raw = json.loads(row_dict.get("decisions_json") or "[]")
                decisions = [Decision.model_validate(dec) for dec in decisions_raw]

                participants = json.loads(row_dict.get("participants_json") or "[]")

                records.append(
                    MemoryRecord(
                        meeting_id=row_dict["meeting_id"],
                        title=row_dict.get("title") or "Executive Meeting",
                        client_id=row_dict.get("client_id"),
                        occurred_at=datetime.fromisoformat(row_dict["occurred_at"]),
                        summary=row_dict["summary"],
                        decisions=decisions,
                        pending_action_items=actions,
                        participants=participants,
                        metadata={},
                    )
                )
            except Exception as e:
                logger.warning("Failed parsing memory record: %s", e)
                continue
        return records

    async def save(self, record: MemoryRecord) -> None:
        async with self._lock:
            actions_json = json.dumps([item.model_dump() for item in record.pending_action_items])
            decisions_json = json.dumps([item.model_dump() for item in record.decisions])
            participants_json = json.dumps(record.participants)
            
            await self.db.execute_commit(
                """
                INSERT INTO meeting_memory (
                    meeting_id, title, client_id, occurred_at, summary,
                    decisions_json, pending_actions_json, participants_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(meeting_id) DO UPDATE SET
                    title = excluded.title,
                    client_id = excluded.client_id,
                    occurred_at = excluded.occurred_at,
                    summary = excluded.summary,
                    decisions_json = excluded.decisions_json,
                    pending_actions_json = excluded.pending_actions_json,
                    participants_json = excluded.participants_json
                """,
                (
                    record.meeting_id,
                    record.title,
                    record.client_id,
                    record.occurred_at.isoformat(),
                    record.summary,
                    decisions_json,
                    actions_json,
                    participants_json,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )


class MongoMemoryStore:
    """Enterprise MongoDB Memory Store for native JSON MOM documents & past intelligence."""

    def __init__(self, uri: str = "mongodb://localhost:27017", database: str = "mom_ai_brain") -> None:
        self.uri = uri
        self.database_name = database
        self._client: Any = None
        self._db: Any = None
        self._connected: bool = False
        self._fallback_sqlite = SQLiteMemoryStore()

    async def is_available(self) -> bool:
        return await self._ensure_connection()

    async def _ensure_connection(self) -> bool:
        if self._connected and self._db is not None:
            return True
        try:
            import motor.motor_asyncio
            self._client = motor.motor_asyncio.AsyncIOMotorClient(self.uri, serverSelectionTimeoutMS=1500)
            await self._client.server_info()
            self._db = self._client[self.database_name]
            self._connected = True
            # Build indexes
            await self._db.meeting_memory.create_index([("occurred_at", -1)])
            await self._db.meeting_memory.create_index([("meeting_id", 1)], unique=True)
            await self._db.action_items_ledger.create_index([("meeting_id", 1), ("owner", 1)])
            return True
        except Exception as ex:
            logger.warning("MongoDB connection unavailable (%s). Using SQLite fallback.", ex)
            self._connected = False
            return False

    async def recall(
        self,
        client_id: str | None,
        limit: int,
    ) -> list[MemoryRecord]:
        if await self._ensure_connection():
            try:
                query = {} if client_id is None else {"client_id": client_id}
                cursor = self._db.meeting_memory.find(query).sort("occurred_at", -1).limit(limit)
                docs = await cursor.to_list(length=limit)
                records: list[MemoryRecord] = []
                for doc in docs:
                    doc.pop("_id", None)
                    records.append(MemoryRecord.model_validate(doc))
                return records
            except Exception as e:
                logger.error("Error querying MongoDB: %s. Using SQLite fallback.", e)
        return await self._fallback_sqlite.recall(client_id, limit)

    async def save(self, record: MemoryRecord) -> None:
        # Save to SQLite durable local store
        await self._fallback_sqlite.save(record)
        if await self._ensure_connection():
            try:
                doc = record.model_dump(mode="json")
                doc["updated_at"] = datetime.now(timezone.utc).isoformat()
                await self._db.meeting_memory.update_one(
                    {"meeting_id": record.meeting_id},
                    {"$set": doc},
                    upsert=True,
                )
                # Store individual action item documents in action_items_ledger collection
                for act in record.pending_action_items:
                    act_doc = act.model_dump(mode="json")
                    act_doc["meeting_id"] = record.meeting_id
                    act_doc["meeting_title"] = record.title
                    act_doc["occurred_at"] = record.occurred_at.isoformat() if isinstance(record.occurred_at, datetime) else str(record.occurred_at)
                    await self._db.action_items_ledger.update_one(
                        {"meeting_id": record.meeting_id, "description": act.description},
                        {"$set": act_doc},
                        upsert=True,
                    )
                logger.info("Successfully persisted meeting '%s' (%s) into MongoDB database '%s'", record.title, record.meeting_id, self.database_name)
            except Exception as e:
                logger.error("Failed to save to MongoDB: %s", e)


class MemoryManager:
    """Manages cross-meeting organizational intelligence and past learning recall."""

    def __init__(self, store: MemoryStore, meeting_limit: int) -> None:
        self._store = store
        self._meeting_limit = meeting_limit

    async def recall(self, contract: M2ToM3Contract) -> list[MemoryRecord]:
        """Recalls the most recent historical meetings for this organizational context."""
        return await self._store.recall(self._client_id(contract), self._meeting_limit)

    async def recall_all(self, limit: int = 20) -> list[MemoryRecord]:
        """Recalls all past stored meetings for organization-wide dashboard intelligence."""
        return await self._store.recall(None, limit)

    @staticmethod
    def format(records: list[MemoryRecord]) -> str:
        """Formats recalled past meetings into an executive organizational memory prompt."""
        if not records:
            return "No previous historical meetings recorded in organizational memory."

        formatted_blocks = []
        for idx, rec in enumerate(records, 1):
            date_str = rec.occurred_at.strftime("%Y-%m-%d") if isinstance(rec.occurred_at, datetime) else str(rec.occurred_at)
            title = rec.title or f"Meeting {rec.meeting_id}"
            
            pending_actions_str = ", ".join(
                f"'{act.description}' (Owner: {act.owner}, Due: {act.deadline_text})"
                for act in rec.pending_action_items[:4]
            ) or "None pending"

            decisions_str = ", ".join(
                f"'{dec.description}'" for dec in rec.decisions[:3]
            ) or "None recorded"

            block = (
                f"Past Meeting #{idx}: \"{title}\" (Date: {date_str})\n"
                f"  • Executive Summary: {rec.summary}\n"
                f"  • Past Approved Decisions: {decisions_str}\n"
                f"  • Ongoing / Pending Action Items: {pending_actions_str}\n"
                f"  • Known Participants: {', '.join(rec.participants) or 'Team'}"
            )
            formatted_blocks.append(block)

        return (
            "=== ORGANIZATIONAL MEMORY & PAST INTELLIGENCE ===\n"
            + "\n\n".join(formatted_blocks)
            + "\n==============================================="
        )

    async def remember(
        self,
        contract: M2ToM3Contract,
        result: MeetingIntelligenceResult,
    ) -> None:
        """Stores the newly completed meeting into long-term organizational memory."""
        title = result.meeting_title or contract.meeting.title or "Executive Strategic Sync"
        await self._store.save(
            MemoryRecord(
                meeting_id=contract.meeting.meeting_id,
                title=title,
                client_id=self._client_id(contract),
                occurred_at=contract.meeting.ended_at,
                summary=result.meeting_summary,
                decisions=result.decisions,
                pending_action_items=[
                    item for item in result.action_items if (getattr(item, "status", "") or "").lower() != "completed"
                ],
                participants=result.participants or [p.display_name for p in contract.meeting.participants],
                metadata=contract.meeting.metadata,
            )
        )
        logger.info(
            "Saved meeting '%s' (%s) into organizational memory with %d pending action items and %d decisions",
            title,
            contract.meeting.meeting_id,
            len(result.action_items),
            len(result.decisions),
        )

    @staticmethod
    def _client_id(contract: M2ToM3Contract) -> str | None:
        value = contract.meeting.metadata.get("client_id")
        return str(value) if value is not None else None


class AIBrainPatternStore:
    """Manages learned AI Brain patterns, domain taxonomies, and agent rules in MongoDB."""

    def __init__(self, uri: str = "mongodb://localhost:27017", database: str = "mom_ai_brain") -> None:
        self.uri = uri
        self.database_name = database
        self._client: Any = None
        self._db: Any = None
        self._connected: bool = False
        self._in_memory: list[dict[str, Any]] = []

    async def _ensure_connection(self) -> bool:
        if self._connected and self._db is not None:
            return True
        try:
            import motor.motor_asyncio
            self._client = motor.motor_asyncio.AsyncIOMotorClient(self.uri, serverSelectionTimeoutMS=1500)
            await self._client.server_info()
            self._db = self._client[self.database_name]
            self._connected = True
            await self._db.ai_brain_patterns.create_index([("agent_type", 1), ("created_at", -1)])
            await self._db.ai_brain_patterns.create_index([("pattern_name", 1)])
            await self._seed_default_patterns()
            return True
        except Exception as ex:
            logger.warning("MongoDB ai_brain_patterns unavailable (%s), using in-memory store.", ex)
            self._connected = False
            return False

    async def _seed_default_patterns(self) -> None:
        """Seeds standard enterprise extraction patterns if collection is empty."""
        try:
            count = await self._db.ai_brain_patterns.count_documents({})
            if count == 0:
                defaults = [
                    {
                        "agent_type": "action",
                        "pattern_name": "SSO & SAML Security Deliverables",
                        "category": "security",
                        "trigger_keywords": ["sso", "saml", "audit", "security", "okta"],
                        "rule_description": "Map security commitments to high priority with verifiable audit sign-off criteria.",
                        "example_output": {"priority": "High", "successCriteria": "Security audit completed and signed off with vulnerability remediation report"},
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                    {
                        "agent_type": "action",
                        "pattern_name": "Database Schema & Cluster Migration",
                        "category": "infrastructure",
                        "trigger_keywords": ["database", "postgres", "migration", "schema", "cluster"],
                        "rule_description": "Map database schema work to Engineering Lead/Database Team with zero data loss criteria.",
                        "example_output": {"priority": "High", "successCriteria": "Migration completed with zero data loss verified on staging"},
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                    {
                        "agent_type": "decision",
                        "pattern_name": "Budget & CapEx Ratification",
                        "category": "governance",
                        "trigger_keywords": ["budget", "approved", "phase 1", "$", "cost"],
                        "rule_description": "Capture exact dollar amounts and approving executive stakeholders.",
                        "example_output": {"description": "Approved $60,000 Phase 1 budget allocation for infrastructure modernization", "approved_by": ["Stakeholders"]},
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                    {
                        "agent_type": "summary",
                        "pattern_name": "Minto Pyramid Executive Synthesis",
                        "category": "executive",
                        "trigger_keywords": ["strategy", "alignment", "kickoff", "milestone"],
                        "rule_description": "Format summary as Situation, Complication, Resolution with 4-6 bullet takeaways.",
                        "example_output": {"confidence": 0.95},
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                    {
                        "agent_type": "risk",
                        "pattern_name": "Rate-Limit & Latency Mitigations",
                        "category": "technical",
                        "trigger_keywords": ["rate limit", "latency", "429", "timeout"],
                        "rule_description": "Recommend Redis token caching and exponential backoff retry.",
                        "example_output": {"mitigation": "Implement Redis token caching layer and asynchronous worker queues"},
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                    {
                        "agent_type": "requirement",
                        "pattern_name": "API & OAuth Performance Specifications",
                        "category": "technical",
                        "trigger_keywords": ["latency", "sub-50ms", "oauth", "throughput"],
                        "rule_description": "Capture quantitative SLA targets and formal architectural requirements.",
                        "example_output": {"category": "technical", "priority": "High"},
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                ]
                await self._db.ai_brain_patterns.insert_many(defaults)
        except Exception as e:
            logger.warning("Could not seed default ai_brain_patterns: %s", e)

    async def list_patterns(self, agent_type: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        if await self._ensure_connection():
            try:
                q = {} if not agent_type or agent_type == "all" else {"agent_type": agent_type}
                cursor = self._db.ai_brain_patterns.find(q).sort("created_at", -1).limit(limit)
                docs = await cursor.to_list(length=limit)
                for d in docs:
                    d["id"] = str(d.pop("_id"))
                return docs
            except Exception as e:
                logger.error("Error listing ai_brain_patterns from MongoDB: %s", e)
        return self._in_memory

    async def save_pattern(self, pattern: dict[str, Any]) -> dict[str, Any]:
        pattern["created_at"] = datetime.now(timezone.utc).isoformat()
        if await self._ensure_connection():
            try:
                res = await self._db.ai_brain_patterns.insert_one(pattern)
                pattern["id"] = str(res.inserted_id)
                pattern.pop("_id", None)
                return pattern
            except Exception as e:
                logger.error("Error saving ai_brain_pattern to MongoDB: %s", e)
        self._in_memory.append(pattern)
        return pattern

    async def delete_pattern(self, pattern_id: str) -> bool:
        if await self._ensure_connection():
            try:
                from bson import ObjectId
                res = await self._db.ai_brain_patterns.delete_one({"_id": ObjectId(pattern_id)})
                return res.deleted_count > 0
            except Exception as e:
                logger.error("Error deleting ai_brain_pattern from MongoDB: %s", e)
        return False

    async def get_stats(self) -> dict[str, Any]:
        if await self._ensure_connection():
            try:
                total_patterns = await self._db.ai_brain_patterns.count_documents({})
                total_golden = await self._db.goldenExamples.count_documents({})
                total_meetings = await self._db.meeting_memory.count_documents({})
                total_actions = await self._db.action_items_ledger.count_documents({})
                
                pipeline = [{"$group": {"_id": "$agent_type", "count": {"$sum": 1}}}]
                agent_counts = {item["_id"]: item["count"] async for item in self._db.ai_brain_patterns.aggregate(pipeline)}
                return {
                    "database": self.database_name,
                    "connected": True,
                    "total_patterns": total_patterns,
                    "total_golden_examples": total_golden,
                    "total_meetings_indexed": total_meetings,
                    "total_ledger_actions": total_actions,
                    "agent_breakdown": agent_counts,
                }
            except Exception as e:
                logger.error("Error getting AI Brain stats from MongoDB: %s", e)
        return {
            "database": self.database_name,
            "connected": False,
            "total_patterns": len(self._in_memory),
            "total_golden_examples": 0,
            "total_meetings_indexed": 0,
            "total_ledger_actions": 0,
            "agent_breakdown": {},
        }

