from mbqa_model import MultiTaskClassifier, MBQAMTEBModelWrapper
import json 
import torch
import os
import mteb

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
with open(os.path.join(dirname, "../checkpoints/CQG-MBQA/questions.json"), "r") as f:
    linear_questions = json.load(f)

model = MultiTaskClassifier(num_labels=len(linear_questions), backbone="WhereIsAI/UAE-Large-V1")

model.load_state_dict(torch.load(os.path.join(dirname, "../checkpoints/CQG-MBQA/multi_task_classifier_uae_3000000.pt"), map_location="cuda:0"))
    
model.to("cuda")

mteb_model = MBQAMTEBModelWrapper(model, linear_questions, is_binary=True, is_sparse=False, binary_threshold=0.5, use_sigmoid=True)


results = evaluation.run(mteb_model, output_folder=os.path.join(dirname, "../results_mteb"))