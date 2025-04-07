def recall_at_k(relevant, retrieved, k):
    retrieved_at_k = set(retrieved[:k])
    if not relevant:
        return 0.0
    return len(retrieved_at_k.intersection(relevant)) / len(relevant)

def precision_at_k(relevant, retrieved, k):
    retrieved_at_k = retrieved[:k]
    if k == 0:
        return 0.0
    num_relevant = sum(1 for item in retrieved_at_k if item in relevant)
    return num_relevant / k

def average_precision(relevant, retrieved, k):
    if not relevant:
        return 0.0
    ap = 0.0
    num_hits = 0
    for i in range(1, k + 1):
        if retrieved[i - 1] in relevant:
            num_hits += 1
            ap += num_hits / i
    return ap / min(len(relevant), k)

def mean_average_precision(queries, predictions, k):
    map_total = 0.0
    num_queries = len(queries)
    for query in queries:
        relevant = set()  # Populate with ground truth relevant assessment IDs for this query
        retrieved = predictions.get(query, [])
        map_total += average_precision(relevant, retrieved, k)
    return map_total / num_queries if num_queries else 0.0
