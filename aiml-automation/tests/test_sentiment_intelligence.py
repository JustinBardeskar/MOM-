import pytest
from app.ai_brain.models import SentimentOutput
from app.ai_brain.quality import ExecutiveSentimentAnalyzer

def test_high_friction_to_alignment_sentiment_analysis():
    transcript = """
    Alex: We are facing severe PostgreSQL write contention under peak load. The current database bottlenecks are completely unacceptable and breaking checkout for enterprise users.
    David: I am very worried about the query latencies. The connection pool is degraded and failing under stress testing.
    Sarah: I hear the concerns, but I have a concrete solution. I will deploy a distributed Redis caching cluster by Friday to offload all session traffic and solve the bottleneck.
    Alex: Fantastic, that sounds like an outstanding plan. If Redis offloads session traffic, that resolves our main vulnerability.
    David: Agreed. In parallel, I will audit the connection pooling and optimize the slow queries by Tuesday.
    Sarah: Confirmed. Everything is on track and I am very confident in our Friday deployment.
    """
    
    sentiment = ExecutiveSentimentAnalyzer.analyze_transcript(transcript)
    
    assert isinstance(sentiment, SentimentOutput)
    assert sentiment.polarity_score > 0.0
    assert sentiment.engagement_level == "High"
    assert "Constructive" in sentiment.overall or "Alignment" in sentiment.overall
    assert len(sentiment.friction_points) >= 1
    assert len(sentiment.alignment_signals) >= 1
    assert "Sarah" in sentiment.speaker_sentiments
    assert "David" in sentiment.speaker_sentiments
    assert "Alex" in sentiment.speaker_sentiments
    assert sentiment.speaker_sentiments["Sarah"] == "Confident & Solution-Oriented"
    assert len(sentiment.chronological_shifts) >= 3


def test_conflict_heavy_sentiment_analysis():
    transcript = """
    Client: The system is crashing constantly and our users are extremely frustrated. This delay is a critical blocker.
    Dev Lead: We cannot afford another outage. The memory leak on Android 12 is causing severe instability.
    Product: We are struggling to reproduce the crash, and the timeline is slipping.
    """
    
    sentiment = ExecutiveSentimentAnalyzer.analyze_transcript(transcript)
    
    assert sentiment.polarity_score < 0.6
    assert len(sentiment.friction_points) >= 2
    assert "Cautious" in sentiment.overall or "Challenging" in sentiment.overall
    assert any("Opening" in s for s in sentiment.chronological_shifts)


def test_enthusiastic_alignment_sentiment_analysis():
    transcript = """
    Elena: The Q3 marketing wireframes look absolutely fantastic. The design agency delivered an outstanding messaging framework.
    Priya: Brilliant! The team is completely on schedule and our webhook retries tested seamlessly.
    Alex: Perfect. I am delighted with the progress and everyone is aligned on the launch date.
    """
    
    sentiment = ExecutiveSentimentAnalyzer.analyze_transcript(transcript)
    
    assert sentiment.polarity_score >= 0.8
    assert len(sentiment.alignment_signals) >= 2
    assert "Collaborative" in sentiment.overall or "Solution-Driven" in sentiment.overall
    assert sentiment.speaker_sentiments.get("Elena") == "Confident & Solution-Oriented"
    assert sentiment.speaker_sentiments.get("Priya") == "Confident & Solution-Oriented"


def test_empty_transcript_safe_fallback():
    sentiment = ExecutiveSentimentAnalyzer.analyze_transcript("")
    assert sentiment.overall == "Constructive & Professional"
    assert sentiment.confidence >= 0.9
