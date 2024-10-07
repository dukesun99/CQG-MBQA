from utils import BagOfTokenEncoder
import os
import mteb

# tasks used in the paper
# tasks = [
#     # STS tasks
#     mteb.get_task("SICK-R", languages = ["eng"]),
#     mteb.get_task("STS12", languages = ["eng"]),
#     mteb.get_task("STS13", languages = ["eng"]),
#     mteb.get_task("STS14", languages = ["eng"]),
#     mteb.get_task("STS15", languages = ["eng"]),
#     mteb.get_task("STS16", languages = ["eng"]),
#     mteb.get_task("STSBenchmark", languages = ["eng"]),
    
#     # Retrieval tasks
#     mteb.get_task("ArguAna", languages = ["eng"]),
#     mteb.get_task("FiQA2018", languages = ["eng"]),
#     mteb.get_task("NFCorpus", languages = ["eng"]),
#     mteb.get_task("SciFact", languages = ["eng"]),
#     mteb.get_task("SCIDOCS", languages = ["eng"]),
    
#     # clustering tasks
#     mteb.get_task("TwentyNewsgroupsClustering", languages = ["eng"]),
#     mteb.get_task("StackExchangeClusteringP2P", languages = ["eng"]),
#     mteb.get_task("BiorxivClusteringP2P", languages = ["eng"]),
#     mteb.get_task("BiorxivClusteringS2S", languages = ["eng"]),
#     mteb.get_task("MedrxivClusteringP2P", languages = ["eng"]),
#     mteb.get_task("MedrxivClusteringS2S", languages = ["eng"]),
#     mteb.get_task("RedditClusteringP2P", languages = ["eng"]),
# ]
# evaluation = mteb.MTEB(tasks=tasks)

benchmark = mteb.get_benchmark("MTEB(eng)")
evaluation = mteb.MTEB(tasks=benchmark)

dirname = os.path.dirname(__file__)

encoder = BagOfTokenEncoder()

results = evaluation.run(encoder, output_folder=os.path.join(dirname, "../results_mteb_bot"))