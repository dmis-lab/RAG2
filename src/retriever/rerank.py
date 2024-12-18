import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def combine_query_evidence(queries, list1, list2, list3, list4, list5):
    evidences_5 = []
    evidences_5 = [sublist1 + sublist2 + sublist3 + sublist4 + sublist5 for sublist1, sublist2, sublist3, sublist4, sublist5 in zip(list1, list2, list3, list4, list5)]
    q_5a_list = []
    for ith, q in tqdm(enumerate(queries)):
        q_5a = []
        for a in evidences_5[ith]:
            q_a = [q, a]
            q_5a.append(q_a)
        q_5a_list.append(q_5a)
    return q_5a_list, evidences_5

def rerank(q_5a_list, evidences_5, top_k):
    device_ids = [2] if torch.cuda.is_available() else None
    tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Cross-Encoder")
    model = AutoModelForSequenceClassification.from_pretrained("ncbi/MedCPT-Cross-Encoder")
    model.eval()
    model = model.to(device_ids[0])

    logits_list = []
    for q_5a in tqdm(q_5a_list):
        with torch.no_grad():
            encoded_q_5a = tokenizer(
                q_5a,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=512,
            )
            encoded_q_5a = {key: tensor.to(device_ids[0]) for key, tensor in encoded_q_5a.items()}
            logits_q_5a = model(**encoded_q_5a).logits.squeeze(dim=1)
            logits_q_5a = logits_q_5a.detach().cpu()
            logits_list.append(logits_q_5a)

    #logits_list_serializable = [tensor.numpy().tolist() for tensor in logits_list]
    #with open('logits_list.json', 'w') as f:
    #    json.dump(logits_list_serializable, f)

    sorted_indices = [sorted(range(len(logits_5)), key=lambda k: logits_5[k], reverse=True) for logits_5 in logits_list]
    top_k_indices = [sorted_indices_i[:top_k] for sorted_indices_i in sorted_indices]
    sorted_evidence_list = []
    for index, data in enumerate(evidences_5):
        sorted_evidence_list.append([data[i] for i in top_k_indices[index]])
        
    return sorted_evidence_list
