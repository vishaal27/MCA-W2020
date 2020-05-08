import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def relevance_feedback(vec_docs, vec_queries, sim, n=10):
    """
    relevance feedback
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """
    rf_sim = sim # change
    vec_queries = vec_queries.todense()
    vec_docs = vec_docs.todense()
    
    for itr in range(3):
        top_rel_docs = np.zeros((len(vec_queries), n, vec_docs.shape[1]))
        top_nonrel_docs = np.zeros((len(vec_queries), n, vec_docs.shape[1]))

        for query_ind in range(len(vec_queries)):
            top_rel_ = np.argsort(-rf_sim[:, query_ind])[:n]
            top_nonrel_=np.argsort(rf_sim[:, query_ind])[:n]

            rel_v = vec_docs[top_rel_]
            nonrel_v = vec_docs[top_nonrel_]

            top_rel_docs[query_ind] = rel_v
            top_nonrel_docs[query_ind] = nonrel_v

        for query_ind in range(len(vec_queries)):
            new_q = vec_queries[query_ind] + 0.5*np.sum(top_rel_docs[query_ind], axis=0)/n - 0.3*np.sum(top_nonrel_docs[query_ind], axis=0)/n
            vec_queries[query_ind] = new_q

        rf_sim = cosine_similarity(vec_docs, vec_queries)

    return rf_sim


def relevance_feedback_exp(vec_docs, vec_queries, sim, tfidf_model, n=10):
    """
    relevance feedback with expanded queries
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        tfidf_model: TfidfVectorizer,
            tf_idf pretrained model
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """

    rf_sim = sim  # change

    vec_queries = vec_queries.todense()
    vec_docs = vec_docs.todense()
    
    for itr in range(3):
        top_rel_docs = np.zeros((len(vec_queries), n, vec_docs.shape[1]))
        top_nonrel_docs = np.zeros((len(vec_queries), n, vec_docs.shape[1]))

        for query_ind in range(len(vec_queries)):
            top_rel_ = np.argsort(-rf_sim[:, query_ind])[:n]
            top_nonrel_=np.argsort(rf_sim[:, query_ind])[:n]

            rel_v = vec_docs[top_rel_]
            nonrel_v = vec_docs[top_nonrel_]

            top_rel_docs[query_ind] = rel_v
            top_nonrel_docs[query_ind] = nonrel_v

        thesaurus = vec_docs.T.dot(vec_docs)

        for query_ind in range(len(vec_queries)):
            new_q = vec_queries[query_ind] + 0.8*np.sum(top_rel_docs[query_ind], axis=0)/n - 0.1*np.sum(top_nonrel_docs[query_ind], axis=0)/n
            vec_queries[query_ind] = new_q

        for query_ind in range(len(vec_queries)):
            top_term_index = np.argmax(vec_queries[query_ind])
            top_term_vec = thesaurus[top_term_index][:, ]
            terms_to_change = np.argsort(-top_term_vec[:, ])[:, :1]

        rf_sim = cosine_similarity(vec_docs, vec_queries)

    return rf_sim