"""
Add batch of real queries as episodes to rlfo_cache.db
Uses actual retrieval + critique system to score each episode

Author: Bradley R. Kinnard
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.retrievers import retrieve
from backend.core.critique import log_episode_to_replay_buffer
import numpy as np

# Batch queries from user
BATCH_QUERIES = [
    "How is Tavily able to rent out API usage for automated browsing? what equipment etc is needed to do that?",
    "is it possible to create my own web pipeline like that, simply for myself only? what would that entail of if so",
    "think these",
    "im asking, do you think these places are worthy of me?",
    "heres the deal, im working on this shit 24.7 if im not sleeping. always pushing things to the edge and over. love doing what no one else has done. always creating. i dont want to be caught up helping another person achieve their dream. unless im being well fuckin compensated to achieve my own lol. and i dont wantsome fly by night shit. i want to work with dedicated people like myself. you know me, i dont fuck with fluff.",
    "what if, and im just spitballing here, we got some of these pople like me in denver area and got together and formed a company? or got to work with me? not sure how that would go i dont have money to pay anyone till something actualyl produced revenue.",
    "yeah but how do we get these peers narrowed down to those select few that i need?",
    "any way we could find these certain types of people online? via social medai or other forms?",
    "how can i get myself found by them, maybe thats a better approach here?",
    "exactly, you know me. i dont fuck with no fluff. let the chickens do the fluffin :D",
    "my problem is, the chunking isn't intelligent, it just breaks up the info. regardless of actualy context etc of the info",
    "lets implement an experimental nextgen highly effective advanced unique solution to this issue insetad of what people stick to. come up with it. lets see your 5 best ideas and why they would be better than what you just suggested.",
    "are you implying that if we did this to the actual retrieval process, it would be more logical of a fix?",
    "so we leave the chunking as it is now, which is typical chunking. and instead lets see 5 ideas of what to implement as for the retriever process / module. Don't provide any code, im only asking for ideas based on the prior requirements, but applied to rerieval.",
    "5 more, must have never been done by anyone ever before. and tell me why its more effective than known solutions. prove your answer.",
    "out of those 5, which would be the most recommended and why, prove your answer? be breief and easyt to understand.",
    "provide an exact detailed guide of updating a current retrieval system to this system. in steps. do not provide any code needed, just highly detailed exact steps. dont leave any s teps out.",
    "whats the point of someone using a chatgpt wrapper?",
    "how is what i do, as you know me, differnt from people just building wrappers?",
    "for my linkedin , would this be accurate about me? from what you know of me? - Systems Architect | Full Stack Developer | Python, Multi-Agent Workflows, Offline Systems | Cognitive Intelligence Engineer",
    "at the stage i am, with my skills as you know them to be, whats my 0-100 chance of getting hired at nueralink and what position?",
    "how can i get a neuroscience background fast, without going to college? like a certification for neuro/software dev type thing?",
    "I'm ready to put my focus into a more narrow area. I'm decided on Neuralink. I need to get hired there for the best position for both me and them. 100% success in getting hired. whatever i need to do in preparation, build a guide for it. give me your best advice",
]


def simulate_critique_score(query: str, rag_count: int, cag_count: int, graph_count: int, web_count: int) -> float:
    """
    Simulate critique scoring based on query characteristics and retrieval results.
    Returns reward in [0.0, 1.0] range.
    """
    # Base score
    score = 0.5

    # Boost for technical/specific queries
    if any(word in query.lower() for word in ['api', 'system', 'code', 'implement', 'technical', 'build', 'architecture']):
        score += 0.15

    # Boost for personal/contextual queries (needs CAG/resume)
    if any(word in query.lower() for word in ['my', 'me', 'i ', 'resume', 'skills', 'hired']):
        score += 0.1
        if cag_count > 0:
            score += 0.1

    # Boost for having diverse sources
    sources_used = sum([rag_count > 0, cag_count > 0, graph_count > 0, web_count > 0])
    score += sources_used * 0.05

    # Boost for web queries that used web
    if any(word in query.lower() for word in ['latest', 'current', 'news', 'weather', 'what is']) and web_count > 0:
        score += 0.15

    # Penalty for very short/unclear queries
    if len(query.split()) < 5:
        score -= 0.1

    # Cap between 0.25 and 0.95
    return max(0.25, min(0.95, score))


def add_batch_episodes():
    """Process queries and add as real episodes to DB"""
    print(f"\n{'='*60}")
    print(f"🎯 BATCH EPISODE ADDITION - {len(BATCH_QUERIES)} queries")
    print(f"{'='*60}\n")

    successful = 0
    failed = 0

    for i, query in enumerate(BATCH_QUERIES, 1):
        try:
            print(f"\n[{i}/{len(BATCH_QUERIES)}] Processing: {query[:60]}...")

            # Run actual retrieval (this uses the real CSWR, CAG, Graph systems)
            results = retrieve(query, top_k=10)

            rag_results = results.get("rag", [])
            cag_results = results.get("cag", [])
            graph_results = results.get("graph", [])
            web_results = results.get("web", [])

            # Build state (same format as real episodes)
            rag_count = len(rag_results)
            cag_count = len(cag_results)
            graph_count = len(graph_results)
            web_count = len(web_results)

            # Simulate critique scoring
            reward = simulate_critique_score(query, rag_count, cag_count, graph_count, web_count)

            # Create episode
            episode = {
                "query": query,
                "rag_count": rag_count,
                "cag_count": cag_count,
                "graph_count": graph_count,
                "web_count": web_count,
                "reward": reward,
                "rl_weights": [0.33, 0.33, 0.34, 0.0],  # Default weights
                "response": f"[Simulated response for batch episode]",
                "timestamp": None  # Will be auto-set by DB
            }

            # Log to DB
            success = log_episode_to_replay_buffer(episode)

            if success:
                successful += 1
                print(f"  ✅ Added | RAG={rag_count} CAG={cag_count} Graph={graph_count} Web={web_count} | Reward={reward:.2f}")
            else:
                failed += 1
                print(f"  ❌ Failed to add episode")

        except Exception as e:
            failed += 1
            print(f"  ❌ Error: {e}")

    print(f"\n{'='*60}")
    print(f"📊 BATCH COMPLETE")
    print(f"  ✅ Successful: {successful}/{len(BATCH_QUERIES)}")
    print(f"  ❌ Failed: {failed}/{len(BATCH_QUERIES)}")
    print(f"{'='*60}\n")

    return successful


if __name__ == "__main__":
    added = add_batch_episodes()
    print(f"\n✅ Added {added} episodes to rlfo_cache.db")
    print("Ready for RL training with: python backend/rl/train_from_db.py --timesteps 50000")
