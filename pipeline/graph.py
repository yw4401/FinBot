from collections import namedtuple
from transformers import T5Tokenizer
import numpy as np
import spacy
import bertopic
from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm
import re


def cosine_sim(a, b):
    return dot(a, b)/(norm(a)*norm(b))


Node = namedtuple("Node", "id doc_id pos text")


nlp = spacy.load("en_core_web_md")
t5_tokenizer = T5Tokenizer.from_pretrained("lmsys/fastchat-t5-3b-v1.0", device_map="auto")
embedding_model = SentenceTransformer('sentence-transformers/LaBSE')


def spacy_sentencizer(text):
    text = re.sub("\s+", " ", text)
    doc = nlp(text)
    return [str(s).strip() for s in doc.sents]


def t5_sentence_weighter(sentence):
    return len(t5_tokenizer.encode(text=sentence, add_special_tokens=False))


def average_cos_embed_sim(sentences, query_embed):
    embeddings = embedding_model.encode(sentences)
    dist_sum = 0
    for i in range(embeddings.shape[0]):
        dist_sum = dist_sum + cosine_sim(embeddings[i, :], query_embed)
    return dist_sum / embeddings.shape[0]


def bertopics_scorer(nodes, query, alpha=0.5, threhsold=0.4):
    sentences = [nodes[n].text for n in nodes]
    topic_model = bertopic.BERTopic(calculate_probabilities=True, embedding_model=embedding_model)
    topics, probabilities = topic_model.fit_transform(sentences)

    id_edges = {i: set() for i in range(np.shape(probabilities)[1])}
    id_weights = {i: 0 for i in id_edges}

    for i in id_edges:
        nids = list((probabilities[:, i] >= threhsold).nonzero()[0])
        id_edges[i].update(nids)

    weight_vec = np.sum(probabilities, axis=0) / len(probabilities)
    for i in id_weights:
        id_weights[i] = weight_vec[i]

    if query:
        query_embedding = embedding_model.encode(query)
        for i in id_weights:
            topic_sents = [sentences[j] for j in id_edges[i]]
            sim = average_cos_embed_sim(topic_sents, query_embedding)
            id_weights[i] = alpha * id_weights[i] + (1 - alpha) * sim
    return id_edges, id_weights


class HyperGraphSummarizer:

    def __init__(self, cluster_scorer, sentence_weighter, sentencizer):
        self.cluster_scorer = cluster_scorer
        self.sentence_weighter = sentence_weighter
        self.sentencizer = sentencizer

    def summarize(self, documents, query=Node, max_weight=1024):
        id_nodes, node_weights = self.create_nodes(documents)
        id_edges, edge_weights = self.cluster_scorer(id_nodes, query)
        result_set = list(self.find_summary_nodes(max_weight, id_nodes, node_weights, id_edges, edge_weights))
        result_set.sort(key=lambda r: id_nodes[r].pos)

        return [id_nodes[r].text for r in result_set]

    def find_summary_nodes(self, max_weight, id_nodes, node_weights, id_edges, edge_weights):
        running_weights = {i: 0 for i in node_weights}
        incident_edges = self.find_incident_edges(id_edges)
        starting_set = list(id_nodes.keys())
        result_set = set()
        final_length = 0

        for i in id_nodes:
            if i not in incident_edges:
                continue
            for e in incident_edges[i]:
                running_weights[i] = running_weights[i] + edge_weights[e]
            running_weights[i] = running_weights[i] / node_weights[i]

        while len(starting_set) != 0:
            max_running_weight = -np.inf
            max_node = -1
            max_idx = -1
            for p, n in enumerate(starting_set):
                if running_weights[n] > max_running_weight:
                    max_running_weight = running_weights[n]
                    max_node = n
                    max_idx = p

            del starting_set[max_idx]
            if node_weights[max_node] + final_length < max_weight:
                result_set.add(max_node)
                final_length = final_length + node_weights[max_node]

                for s in starting_set:
                    if s not in running_weights:
                        continue
                    max_incident = incident_edges[max_node] if max_node in incident_edges else set()
                    s_incident = incident_edges[s] if s in incident_edges else set()
                    adj_set = max_incident.intersection(s_incident)
                    adj_weight = 0
                    for e in adj_set:
                        if e not in edge_weights:
                            continue
                        adj_weight = adj_weight + edge_weights[e]
                    adj_weight = adj_weight / node_weights[s]
                    running_weights[s] = running_weights[s] - adj_weight
        return result_set

    def find_incident_edges(self, id_edges):
        incident_edges = {}
        for e in id_edges:
            nids = id_edges[e]
            for n in nids:
                if n not in incident_edges:
                    incident_edges[n] = {e}
                else:
                    incident_edges[n].add(e)
        return incident_edges

    def create_nodes(self, documents):
        doc_id = 0
        node_id = 0
        id_nodes = {}
        node_weights = {}
        for d in documents:
            sentences = self.sentencizer(d)
            for pos, s in enumerate(sentences):
                s_len = self.sentence_weighter(s)
                s_nodes = Node(id=node_id, doc_id=doc_id, pos=pos, text=s)
                id_nodes[node_id] = s_nodes
                node_weights[node_id] = s_len
                node_id = node_id + 1
            doc_id = doc_id + 1

        return id_nodes, node_weights


if __name__ == "__main__":

    test_article = """Seniors with early Alzheimer’s disease will face major hurdles to get treated even if promising new drugs roll out more broadly in the coming years, putting them at risk of developing more severe disease as they wait months or perhaps years for a diagnosis.

    The U.S. health-care system is not currently prepared to meet the needs of an aging population in which a growing number of people will need to undergo evaluation for Alzheimer’s, according to neurologists, health policy experts and the companies developing the drugs.

    There are not enough dementia specialists or the needed testing capacity in the U.S. to diagnose everyone who may benefit from a new treatment like Eisai
     and Biogen
    ’s Leqembi. After patients are diagnosed, the capacity may not exist — at least initially — to provide the twice monthly intravenous infusions for everyone who is eligible.

    Researchers estimate that the wait time from the initial evaluation to the confirmatory diagnostic tests to the infusions could range anywhere from a year and a half to four years or longer. Those months are critical for people with Alzheimer’s.

    “The whole process from that time of the family physician conversation to the point of infusion, I worry how long it will take and the complexities of the patient navigating through all of that to successfully get to the end,” Anne White, president of neuroscience at Eli Lilly
    , which is developing its own Alzheimer’s treatment, told CNBC.

    There are promising innovations in development, such as blood tests and injections that patients would take at home, which could make it significantly easier to get diagnosed and treated in the future.

    White also said Lilly is confident that more doctors will get into the field and help to alleviate capacity issues, as awareness grows that medicines are entering the market to treat Alzheimer’s.

    But time spent waiting robs early patients of their memory and ability to live independently. Alzheimer’s gets worse with time, and as patients deteriorate into more advanced stages of the disease, they no longer benefit from treatments like Leqembi that are designed to slow cognitive decline early.

    More than 2,000 seniors transition from mild to moderate dementia from the disease a day, according to estimates from the Alzheimer’s Association. At that stage, they become ineligible for Leqembi.

    The central challenge is that a large and rapidly growing group of people have early memory loss and other thinking problems known as mild cognitive impairment. This condition is often, though not always, a sign of early Alzheimer’s disease.

    An estimated 13 million people in the U.S. had mild cognitive impairment last year, according to a study published in the Alzheimer’s and Dementia Journal. As the U.S. population ages, the number of people with this condition is expected to reach 21 million by 2060, the study projected.

    The U.S. health-care system will deal with major logistical challenges in diagnosing the growing population of people with early Alzheimer’s — even before patients face potential issues with accessing treatment.

    “There’s a very large population of undiagnosed cognitive impairments that need to be evaluated in order to determine if people are eligible for treatment,” said Jodi Liu, an expert on health policy at the Rand Corporation. Access to drugs like Leqembi is severely restricted because Medicare for now will only cover the $26,500-per-year treatment for people participating in clinical trials. Medicare has promised to provide broader coverage if Leqembi receives full approval from the Food and Drug Administration, which Eisai expects to happen in July.

    Eisai has estimated that 100,000 people in the U.S. will be diagnosed and eligible for Leqembi by the third year of the treatment’s rollout. The sum is a fraction of the total population that could benefit.

    Those patients could have other options if new treatments emerge from trials with positive marks.

    Eli Lilly will publish clinical trial data on its antibody infusion donanemab in the second quarter of this year. If the data is positive, the company will ask the FDA to approve the drug.

    Eisai’s U.S. CEO Ivan Cheung and Lilly’s White said during the companies’ respective earnings calls in February that they are focused on working with the U.S. health system to address the challenges of rolling out of Alzheimer’s treatments.

    “The primary goal right now during this launch phase [...] is really get the market ready in terms of the diagnostic pathway, the infusion capacity, the education on how to monitor for this therapy, get all the hospitals and clinics ready,” Cheung said.

    Not enough specialists
    Long lines are expected at the offices of geriatricians, neurologists and radiologists as millions of people with mild cognitive impairment undergo evaluation to diagnose whether they have Alzheimer’s disease.

    Demand for geriatricians — doctors who are experts in diseases that affect the elderly — is expected to outstrip the number of specialists available in the field through at least 2035, according to projections from the federal Health Resources and Services Administration.

    The American Academy of Neurology told Medicare in a February letter that increased demand for Alzheimer’s treatments will put substantial pressure on neurologists, who will need additional resources. The federal data predicts a substantial shortage of these specialists in rural areas through at least 2035.

    “You just look at the neurologists, look at geriatricians — there are fewer and fewer geriatricians per person in the U.S.,” Rand’s Liu said. “It’s just a few number of specialists to do this kind of work.”

    White said Lilly has heard stories of patients waiting six to 12 months to see a neurologist or other doctors who treat dementia due to current capacity issues.

    The number of radiologists — who also play a role in diagnosing the disease — is expected to decline in the U.S. through 2035 even as demand increases, the data shows.

    In a study published in 2017, Liu and other Rand researchers estimated an initial wait of 18 months for patients to get evaluated by a dementia specialist, tested to confirm a diagnosis, and then treated in the first year that an Alzheimer’s antibody treatment becomes available. The wait would decrease to 1.3 months by 2030 as the patient backlog is cleared, they estimated at the time.

    But more recent research found that the wait would actually increase as demand created by an aging U.S. population outstrips the supply of specialists.

    Patients seeking a first specialist visit could face an initial wait of 20 months, according to a study by researchers at the University of Southern California published in the journal Alzheimer’s and Dementia in 2021. The delay could increase to about four years as early as 2028 and grow longer through 2050, the study found.

    The journal is published by the Alzheimer’s Association.

    Both studies are based on assumptions made before Leqembi received expedited approval from the FDA in January. Actual wait times could differ from the studies’ projections.

    PET scans cumbersome
    Two types of tests can diagnosis Alzheimer’s disease: PET scans and spinal taps. PET scans are accurate and safe diagnostic tools, but they are also cumbersome and expensive, said Dr. David Russell, a neurologist.

    Patients are injected with a tracer that makes brain abnormalities visible to the machine that does the imaging. Tracers have to be made for each patient and used on the same day.

    “We don’t have the infrastructure to roll out PET scanning on a major scale,” said Russell, director of clinical research at the Institute for Neurodegenerative Disorders in New Haven, Connecticut. He is the principal investigator on the clinical trials of Leqembi and donanemab at the institute.

    Medicare coverage of PET scans for Alzheimer’s patients is also limited right now. The insurance program for seniors will only cover one scan per lifetime, and only when the patient is participating in a clinical trail approved by the federal Centers for for Medicare and Medicaid Services.

    “That’s concerning because people may actually test negative at one point but then obviously as they age, they may need to get tested again,” White said.

    Early Alzheimer’s disease can also be diagnosed with a spinal tap, in which fluid around the spinal cord is extracted with a catheter and tested. While there’s plenty of capacity to do spinal taps, this option isn’t attractive to many patients due to unfounded fears that it’s painful and dangerous, Russell said.

    Though “there’s a lot of resistance” to the procedure, it is well tolerated and safe, he noted.

    Rural areas at a disadvantage
    The lack of access to PET scans is even more of an issue for patients who live in rural areas.

    There are an estimated 2,300 PET scan machines in the U.S., according to a 2021 study published in Alzheimer’s and Dementia. But the machines are often in bigger cities, which puts people in rural areas at a disadvantage.

    “There are certainly areas that don’t have a PET scanner, rural areas, so people would need to travel to a health center that has a PET scanner,” Liu said.

    In a large, sparsely populated rural state like New Mexico, many patients would have to drive three to five hours to get a PET scan in a city such as Albuquerque, said Dr. Gary Rosenberg, a neurologist and director of the New Mexico Alzheimer’s Disease Research Center.

    “It’s not California or the East Coast where everything’s very compressed and people can travel and get to a center pretty easily and go through these kinds of treatments,” Rosenberg said.

    The state has an estimated population of 43,000 people with dementia, and there are very few neurologists outside of the Albuquerque area, Rosenberg said. The New Mexico Alzheimer’s Disease Research Center in Albuquerque is one of only three such facilities funded by the federal National Institute of Aging in a vast region stretching west from Texas to Arizona.

    To do a PET scan, a tracer has to be made for each patient off-site in Phoenix, flown on a private plane to Albuquerque and used within hours because the tracers have a short shelf life, according to Rosenberg. The whole process costs more than $12,000 per patient, he added.

    “It’s logistically going to be very challenging,” Rosenberg said.

    IV infusion capacity
    After spending months or possibly years waiting to get diagnosed with early Alzheimer’s, patients would then be eligible for intravenous infusions of Leqembi. But the U.S. doesn’t currently have the capacity to give infusions twice monthly for everyone who likely has the disease, Russell said.

    “Having an IV infusion every two weeks would sort of ration people to availability and that’s a problem,” Russell said.

    The University of New Mexico Hospital is already maxed out with demand for infusion therapies for cancer, rheumatoid arthritis and autoimmune diseases, and could have a “problem” adding new capacity, said Rosenberg.

    Intravenous infusions of monoclonal antibodies like Leqembi aren’t difficult to administer, Russell said.

    The infrastructure to offer infusions should expand rapidly once industry sees there’s demand for treatments like Leqembi. But the process of building out capacity could still take a couple years, Russell said. He believes big players like CVS
     will provide infusions for Alzheimer’s disease on a major scale if they see there’s a large and stable market.

    “In one sense, capitalism works, and if it looks like that’s going to be the future, I think infusion centers will explode onto the scene,” the neurologist said.

    Eisai and Biogen hope to move early Alzheimer’s patients to a single monthly dose of Leqembi after they’ve completed their initial course of twice monthly infusions, which could help alleviate some of the capacity issues with infusions over time. They plan to ask the FDA to approve this plan in early 2024.

    Eli Lilly’s Alzheimer’s candidate antibody treatment donanemab is a single monthly dose, potentially making the logistics of administration easier if the drug gets approved. Dr. Dan Skovronsky, Lilly’s chief medical officer, told analysts during the company’s first-quarter earnings call that he expects many patients will be able to stop taking donanemab at 12 months.

    Blood tests could reduce wait times
    Though the projected wait times to get diagnosed and treated are sobering, innovations on the horizon promise to significantly improve access to Alzheimer’s drugs over time.

    Blood tests for Alzheimer’s are in development and some are already on the market. Primary-care doctors could administer the tests, which would ease the burden on patients, especially those in rural communities where the closest PET scan machine is hours away.

    These tests detect proteins in the blood associated with Alzheimer’s. They promise to help diagnose the disease before people display cognitive symptoms, potentially giving patients the chance to get treated before they suffer irreparable brain damage, according to the National Institutes of Health.

    At least three blood tests made by C2N Diagnostics, Quest Diagnostics
     and Quanterix
     are currently on the market. But they are used to evaluate people who are already presenting symptoms and aren’t available on the mass scale needed for the expected increase in Alzheimer’s patients.

    C2N’s PrecivityAD test costs $1,250 and is not covered by insurance — though the company has a financial assistance program. Quest Diagnostics’ AD-Detect test costs $650. Quest’s test is covered by some insurance plans but not Medicare at the moment. The company also has a financial assistance program. Quanterix wouldn’t disclose the price of its test, which insurance does not cover.

    Right now, these are not stand-alone tests that can definitively diagnose Alzheimer’s. But the tests could help identify the patients who likely have the disease, which would narrow the population that needs further evaluation and reduce wait times for dementia specialists or confirmatory PET scans.

    A study in the journal Alzheimer’s and Dementia estimated that a cognitive test combined with a blood test could slash wait times for dementia specialists from 50 months down to 12 months.

    Eisai believes that inexpensive blood tests could completely replace PET scans and spinal taps by the fourth year of Leqembi’s rollout. The quicker diagnosis could increase the number of people eligible for treatment.

    Rosenberg said widespread availability of blood tests will allow mobile clinics to go into rural communities and identify who has markers associated with Alzheimer’s. This would allow patients in remote towns avoid the hours-long drive to cities with PET scan machines, Rosenberg said.

    “It’s a game changer,” the neurologist said.

    Lilly is developing at least two blood tests. The company is already using one test in clinical trials and hopes to commercialize it sometime this year. It is developing a second test through a collaboration with Roche. White said it is reasonable to expect that in a few years blood tests could replace more burdensome PET scans.

    Injections could make treatment easier
    Biogen and Eisai are also developing an injectable form of Leqembi which patients could administer themselves with an autoinjector similar to insulin pens, saving the trip to a site that provides intravenous infusions. They plan to ask the FDA to approve these so-called subcutaneous injections in early 2024.

    Eli Lilly is also conducting clinical trials on an antibody treatment called remternetug as a self-administered injection. But the promise of injections that can be administered at home could make companies reluctant to invest in building out intravenous infusion capacity, Russell said.

    In the future, Alzheimer’s diagnosis and treatment could be folded into routine checkups with a family doctor, Russell said. When people turn 50 and head in to get a colonoscopy or a cholesterol check, the doctor could also run a blood test for Alzheimer’s.

    If the test comes back positive, the doctor could then schedule patients for an MRI and get them started on an autoinjector treatment, Russell said.

    “That’s going to be the way that we’re looking at it in the not too distant future,” he said."""

    summarizer = HyperGraphSummarizer(bertopics_scorer, t5_sentence_weighter, spacy_sentencizer)
    result = summarizer.summarize([test_article], "Which company makes Leqembi?", max_weight=800)
    for r in result:
        print(r)
