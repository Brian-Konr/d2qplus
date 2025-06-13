from utils.data import combine_topic_info

topic_dir = "/home/guest/r12922050/GitHub/d2qplus/augmented-data/nfcorpus/topics/0612-01"
corpus_path = "/home/guest/r12922050/GitHub/d2qplus/data/nfcorpus/corpus.jsonl"
def visualize_topic_keywords():

    corpus = combine_topic_info(
        enhanced_topic_info_pkl=topic_dir + "/topic_info_dataframe_enhanced.pkl",
        corpus_topics_path=topic_dir + "/doc_topics.jsonl",
        core_phrase_path=topic_dir + "/keywords.jsonl",
        corpus_path=corpus_path,
    )

    # randomly pick 3 documents from corpus list
    import random
    random_docs = random.sample(corpus, 3)

    with open(f"{topic_dir}/example_docs.md", "w") as f:
        for doc in random_docs:
            f.write(f"## Document ID: {doc['doc_id']}\n")
            f.write(f"### Title: {doc['title']}\n")
            f.write(f"### Text: \n{doc['text']}\n\n")

            # print topic and topic keywords
            # sort topics by weight

            f.write("### Topics:\n")
            sorted_topics = sorted(doc['topics'], key=lambda x: x['weight'], reverse=True)
            for topic in sorted_topics[:3]:
                f.write(f"#### Topic ({topic['topic_id']}): {topic['Enhanced_Topic']} ({topic['weight']})\n")
                f.write(f"Keywords: {', '.join(topic['Representation'])}\n\n")
            f.write("\n---\n\n")

            # print core phrases
            f.write("### Core Phrases:\n")
            f.write(", ".join([kw['phrase'] for kw in doc['core_phrases']]) + "\n")

            f.write("\n---\n\n")

    print("Example documents written to example_docs.md")

if __name__ == "__main__":
    visualize_topic_keywords()
    # visualize_topic_keywords(topic_dir, corpus_path)
    # visualize_topic_keywords("/home/guest/r12922050/GitHub/d2qplus/augmented-data/nfcorpus/topics/0612-01", "/home/guest/r12922050/GitHub/d2qplus/data/nfcorpus/corpus.jsonl")