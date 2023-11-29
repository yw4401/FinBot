# For data manipulation
import pandas as pd

# Regular expressions, hashing, mathematical operations, time
import re, hashlib, math, time

# NetworkX for graph operations
import networkx as nx

# Random number generation
from random import randint, seed

# Configuration settings
import config

seed(1631996)


# Class defining a hash family for hashing elements
class hashFamily:
    def __init__(self, i):
        self.resultSize = 8  # how many bytes we want back
        self.maxLen = 20  # how long can our i be (in decimal)
        self.salt = str(i).zfill(self.maxLen)[-self.maxLen:]

    def get_hash_value(self, el_to_hash):
        return int(
            hashlib.sha1(str(el_to_hash).encode('utf-8') + self.salt.encode('utf-8')).hexdigest()[-self.resultSize:],
            16)


# Class defining a shingler for text shingling operations
class shingler:
    def __init__(self, k):

        if k > 0:
            self.k = int(k)
        else:
            self.k = 10

    # inner class utility
    # Method to process the document text
    def process_doc(self, document):
        return re.sub("( )+|(\n)+", " ", document).lower()

    # Method to extract shingles from the document
    def get_shingles(self, document):
        shingles = set()
        document = self.process_doc(document)
        tokens = document.split()  # Split the document into tokens
        for i in range(0, len(tokens) - self.k + 1):
            shingle = " ".join(tokens[i:i + self.k])
            shingles.add(shingle)
        return shingles

    def get_k(self):
        return self.k

    
    # Method to get the sorted hashed shingles
    def get_hashed_shingles(self, shingles_set):
        hash_function = hashFamily(0)
        return sorted({hash_function.get_hash_value(s) for s in shingles_set})


# Class defining a minhash signer for generating signatures
class minhashSigner:
    def __init__(self, sig_size):
        self.sig_size = sig_size
        self.hash_functions = [hashFamily(randint(0, 10000000000)) for i in range(0, sig_size)]

    def compute_set_signature(self, set_):
        set_sig = []
        for h_funct in self.hash_functions:
            min_hash = math.inf
            for el in set_:
                h = h_funct.get_hash_value(el)
                if h < min_hash:
                    min_hash = h

            set_sig.append(min_hash)

        return set_sig

    # return a list of lists that can be seen as the signature matrix
    def compute_signature_matrix(self, set_list):
        signatures = []
        # print(len(signatures))
        # print("----------------------------------------")
        # print("set_list: ",len(set_list),set_list)
        for s in set_list:
            signatures.append(self.compute_set_signature(s))

        return signatures


class lsh:
    def __init__(self, threshold=0.8):
        self.threshold = threshold

    def get_signature_matrix_bands(self, sig_matrix, bands_nr, sign_len):
        # bands_nr = b
        # sign_len = n
        r = int(sign_len / bands_nr)  # number of rows in each band
        bands = {}  # {band_nr: [col_1,col_2,...]} where col_1 is all the values of Sig(S_i) for band b.
        for i in range(0, bands_nr):
            bands[i] = []

        # put Subsets of the columns of signature matrix into the appropriate bucket and cosider a column 
        # as a unique block so that we can hash the entire column.
        # Basically a band is a list of element, where each element is a subset of a signature of a given set.
        for signature in sig_matrix:

            for i in range(0, bands_nr):
                idx = i * r
                bands[i].append(' '.join(str(x) for x in signature[idx:idx + r]))

        return bands

    # band is a list
    # construct a dictionary {hash(band_column): doc_id that produced this hash}
    def get_band_buckets(self, band, hash_funct):
        buckets = {}
        for doc_id in range(0, len(band)):
            value = hash_funct.get_hash_value(band[doc_id])
            if value not in buckets:
                buckets[value] = [doc_id]
            else:
                buckets[value].append(doc_id)

        return buckets

    def get_candidates_list(self, buckets):
        candidates = set()
        # buckets is a dictionary containing key=bucket, value= list of doc_ids that hashed to bucket
        for bucket, candidate_list in buckets.items():
            if len(candidate_list) > 1:
                for i in range(0, len(candidate_list) - 1):
                    for j in range(i + 1, len(candidate_list)):
                        pair = tuple(sorted((candidate_list[i], candidate_list[j])))
                        candidates.add(pair)

        return candidates  # ie a set of couples, each couple is a candidate pair

    def check_candidates(self, candidates_list, threshold, sigs):
        similar_docs = set()  # set of tuples

        # similar_pair is a couple containing doc_ids of documents that hashed to same bucket
        for similar_pair in candidates_list:
            # for all the pairs of document in the list check similarity of their signatures
            doc_id_1 = similar_pair[0]
            doc_id_2 = similar_pair[1]
            signature_1 = set(
                sigs[doc_id_1])  # get the i-th column from signature matrix where i is doc_id in the collision list
            signature_2 = set(sigs[doc_id_2])
            js = len(signature_1.intersection(signature_2)) / len(signature_1.union(signature_2))

            if js >= threshold:
                similar_docs.add(tuple(sorted((doc_id_1, doc_id_2))))

        return similar_docs

    def get_similar_items(self, sig_matrix, bands_nr, sign_len):
        similar_docs = set()
        # divide signature matrix into bands
        bands = self.get_signature_matrix_bands(sig_matrix, bands_nr, sign_len)

        # for all the bands
        for band_id, elements in bands.items():
            # produce the buckets for the given band (band_id) with a random hash function
            buckets = self.get_band_buckets(elements, hash_funct=hashFamily(randint(0, 10000000000)))
            # Get all the candidate pairs
            candidates = self.get_candidates_list(buckets)
            # Check all candidate pairs' signatures
            for sim_tuple in self.check_candidates(candidates, self.threshold, sig_matrix):
                similar_docs.add(sim_tuple)

        return similar_docs  # return all the similar signatures that respect the threshold


def get_duplicate_nodes(lsh_similar_itemset):
    # Create the graph (as in the previous example)
    G = nx.Graph()

    for pair in lsh_similar_itemset:
        for document in pair:
            if not G.has_node(document):
                G.add_node(document)
        G.add_edge(pair[0], pair[1])

    components = 0
    components_map = {}

    for node in G.nodes():
        if node not in components_map:
            connected_component = list(nx.node_connected_component(G, node))
            if not connected_component or len(connected_component) == 0:
                connected_component = [node]
            for n in connected_component:
                components_map[n] = components
            components = components + 1

    return components_map, components


def execute_deduplication(src_df: pd.DataFrame):
    # Remove duplicates based on columns: source, category, title and body
    src_df = src_df.sort_values(by="exists", ascending=False)
    src_df = src_df.drop_duplicates(subset=['body'], keep="first").copy()
    src_df.reset_index(drop=True, inplace=True)

    src_df['doc_id'] = src_df.index
    doc_nr = src_df['doc_id'].max()

    print("Dataset loaded correctly.")
    print("Producing Shingles...")
    start_time = time.time()

    shingling_list = [None] * (doc_nr + 1)
    shingling_size = 10
    signature_size = 50
    bands_nr = 10

    shingler_inst = shingler(shingling_size)
    signer = minhashSigner(signature_size)

    # produce hashed shinglings for all documents
    for index, row in src_df.iterrows():
        doc = row['title'] + " " + row['body']
        i = row['doc_id']

        shinglings = shingler_inst.get_hashed_shingles(shingler_inst.get_shingles(doc))
        shingling_list[i] = shinglings

    end_time = time.time()
    print("Shingles produced in:\t %.2f seconds." % (end_time - start_time))

    start_time = time.time()
    print("Computing signature matrix...")
    # produce a signature for each shingle set
    signature_matrix = signer.compute_signature_matrix(shingling_list)
    end_time = time.time()
    print("Signature Matrix computed in:\t %.2f seconds." % (end_time - start_time))

    lsh_instance = lsh(threshold=0.5)
    start_time = time.time()
    print("Computing LSH similarity...")

    print(isinstance(lsh_instance, lsh))
    lsh_similar_itemset = lsh_instance.get_similar_items(signature_matrix, bands_nr, signature_size)
    end_time = time.time()
    lsh_computation_time = end_time - start_time
    print("LSH Similarity computed in:\t %.2f seconds.\nSimilar Elements Found: %d" % (
    lsh_computation_time, len(lsh_similar_itemset)))

    def get_component(component_map, max_component, node):
        if node not in component_map:
            component_map[node] = max_component[0]
            max_component[0] += 1
        return component_map[node]

    # Remove duplicates from the graph and get the list of removed nodes
    component_map, components = get_duplicate_nodes(lsh_similar_itemset)
    max_comp = [components]
    src_df["component"] = src_df.doc_id.apply(lambda idx: get_component(component_map, max_comp, idx))
    num_unique = len(src_df.component.unique())
    print("Number of duplicates: ", src_df.shape[0] - num_unique)

    src_df = src_df.sort_values(by="exists", ascending=False)
    src_df = src_df.drop_duplicates(subset=['component'], keep="first")

    return src_df
