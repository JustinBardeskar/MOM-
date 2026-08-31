# Meeting Taxonomy & Classification Guide

When the Meeting Understanding Agent classifies a meeting, it must select exactly ONE of the following 10 canonical meeting types based on dominant structural discussion, not single substring matches.

---

## 1. `technical`
- **Focus**: Architecture, APIs, database schemas, code refactoring, performance optimization, security, system latency.
- **Indicators**: `API`, `schema`, `database`, `backend`, `frontend`, `endpoint`, `latency`, `cluster`, `docker`, `codebase`, `deployment`, `microservices`.

## 2. `scrum`
- **Focus**: Sprint cadence, standup sync, task estimation, story points, backlog grooming, retrospective.
- **Indicators**: `sprint`, `standup`, `blocker`, `story points`, `retro`, `daily sync`, `ticket`, `kanban`, `burndown`.

## 3. `product`
- **Focus**: UI/UX design, wireframes, user journeys, feature roadmaps, customer onboarding, release scope.
- **Indicators**: `figma`, `wireframes`, `user experience`, `onboarding funnel`, `user story`, `design system`, `feature spec`.

## 4. `marketing`
- **Focus**: Campaigns, social media, go-to-market (GTM) launches, brand messaging, press releases.
- **Indicators**: `campaign`, `linkedin`, `product launch`, `gtm`, `messaging framework`, `social media`, `press release`, `brand collateral`.

## 5. `sales`
- **Focus**: Pricing tiers, sales quotas, customer contract negotiations, pipeline, deals, client pitches.
- **Indicators**: `pricing tier`, `annual contract`, `arr`, `mrr`, `pipeline`, `quota`, `discount`, `deal closing`, `sales enablement`.

## 6. `client`
- **Focus**: External client/customer sync, Statement of Work (SOW), Service Level Agreements (SLA), customer deliverables.
- **Indicators**: `client sync`, `customer milestone`, `sow`, `sla`, `vendor review`, `client onboarding`.

## 7. `strategy`
- **Focus**: Quarterly business reviews (QBR), executive board reviews, annual budgeting, high-level OKRs.
- **Indicators**: `qbr`, `okr`, `strategic priority`, `annual budget`, `executive review`, `board meeting`, `governance`.

## 8. `interview`
- **Focus**: Candidate assessment, resume evaluation, technical coding challenge, behavioral evaluation.
- **Indicators**: `candidate`, `resume`, `interview`, `hiring`, `tell me about yourself`, `coding challenge`.

## 9. `hr`
- **Focus**: Internal human resources, annual performance appraisals, 1-on-1 career sync, compensation, employee leave policy.
- **CRITICAL**: Use whole-word matching (`\bhr\b`, `"human resources"`). **NEVER** classify as HR merely because words like `"three"`, `"threat"`, or `"through"` contain the letters "hr".

## 10. `general`
- **Focus**: Multi-purpose informational sync or team alignment that does not fit a specialized technical/operational archetype.
