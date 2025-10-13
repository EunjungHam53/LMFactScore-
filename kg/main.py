# import os
# import json
# from pathlib import Path

# from .preprocess import preprocess
# from .save_triples import save_triples_for_scripts
# from .generate_kg import init_kg, refine_kg

# def script2kg(scene_list, idx, args, name):

#     # 1) preprocess script
#     preprocessed_script = preprocess(scene_list, idx)
#     save_root = Path(args.kg_path) / f'{idx}_{name}'
#     preprocess_path = Path(save_root) / '1_preprocessed/script.json'
#     os.makedirs(preprocess_path .parent, exist_ok=True)
    
#     with open(preprocess_path , 'w') as f:
#         json.dump(preprocessed_script, f, indent=4, ensure_ascii=False)
    
#     # 2) extract triples
#     extract_path = Path(save_root) / '2_responses'
#     if not args.response_exist or not os.path.exists(extract_path):
#         save_triples_for_scripts(preprocessed_script, idx, extract_path)

#     # 3) build kg
#     save_kg_path = Path(save_root) / '3_knowledge_graphs/'
#     init_kg(idx, save_root, save_kg_path)

#     # 4) refine kg
#     refine_kg(idx, save_root, topk=10, refine=args.refine)