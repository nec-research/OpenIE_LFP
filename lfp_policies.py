#        Improving Cross-Lingual Transfer for Open Information Extraction with Linguistic Feature Projection 
	  
#   File:     lfp_policies.py
#   Authors:  Youmi Ma (youmi.ma@nlp.c.titech.ac.jp) - creator of the code
#             Carolin Lawrence (carolin.lawrence@neclab.eu) - NEC contact

# NEC Laboratories Europe GmbH, Copyright (c) 2023, All rights reserved.  

#        THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
 
#        PROPRIETARY INFORMATION ---  

# SOFTWARE LICENSE AGREEMENT

# ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY

# BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
# LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
# DOWNLOAD THE SOFTWARE.

# This is a license agreement ("Agreement") between your academic institution
# or non-profit organization or self (called "Licensee" or "You" in this
# Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
# Agreement).  All rights not specifically granted to you in this Agreement
# are reserved for Licensor. 

# RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
# ownership of any copy of the Software (as defined below) licensed under this
# Agreement and hereby grants to Licensee a personal, non-exclusive,
# non-transferable license to use the Software for noncommercial research
# purposes, without the right to sublicense, pursuant to the terms and
# conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
# LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
# Agreement, the term "Software" means (i) the actual copy of all or any
# portion of code for program routines made accessible to Licensee by Licensor
# pursuant to this Agreement, inclusive of backups, updates, and/or merged
# copies permitted hereunder or subsequently supplied by Licensor,  including
# all or any file structures, programming instructions, user interfaces and
# screen formats and sequences as well as any and all documentation and
# instructions related to it, and (ii) all or any derivatives and/or
# modifications created or made by You to any of the items specified in (i).

# CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
# proprietary to Licensor, and as such, Licensee agrees to receive all such
# materials and to use the Software only in accordance with the terms of this
# Agreement.  Licensee agrees to use reasonable effort to protect the Software
# from unauthorized use, reproduction, distribution, or publication. All
# publication materials mentioning features or use of this software must
# explicitly include an acknowledgement the software was developed by NEC
# Laboratories Europe GmbH.

# COPYRIGHT: The Software is owned by Licensor.  

# PERMITTED USES:  The Software may be used for your own noncommercial
# internal research purposes. You understand and agree that Licensor is not
# obligated to implement any suggestions and/or feedback you might provide
# regarding the Software, but to the extent Licensor does so, you are not
# entitled to any compensation related thereto.

# DERIVATIVES: You may create derivatives of or make modifications to the
# Software, however, You agree that all and any such derivatives and
# modifications will be owned by Licensor and become a part of the Software
# licensed to You under this Agreement.  You may only use such derivatives and
# modifications for your own noncommercial internal research purposes, and you
# may not otherwise use, distribute or copy such derivatives and modifications
# in violation of this Agreement.

# BACKUPS:  If Licensee is an organization, it may make that number of copies
# of the Software necessary for internal noncommercial use at a single site
# within its organization provided that all information appearing in or on the
# original labels, including the copyright and trademark notices are copied
# onto the labels of the copies.

# USES NOT PERMITTED:  You may not distribute, copy or use the Software except
# as explicitly permitted herein. Licensee has not been granted any trademark
# license as part of this Agreement.  Neither the name of NEC Laboratories
# Europe GmbH nor the names of its contributors may be used to endorse or
# promote products derived from this Software without specific prior written
# permission.

# You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
# whole or in part, or provide third parties access to prior or present
# versions (or any parts thereof) of the Software.

# ASSIGNMENT: You may not assign this Agreement or your rights hereunder
# without the prior written consent of Licensor. Any attempted assignment
# without such consent shall be null and void.

# TERM: The term of the license granted by this Agreement is from Licensee's
# acceptance of this Agreement by downloading the Software or by using the
# Software until terminated as provided below.  

# The Agreement automatically terminates without notice if you fail to comply
# with any provision of this Agreement.  Licensee may terminate this Agreement
# by ceasing using the Software.  Upon any termination of this Agreement,
# Licensee will delete any and all copies of the Software. You agree that all
# provisions which operate to protect the proprietary rights of Licensor shall
# remain in force should breach occur and that the obligation of
# confidentiality described in this Agreement is binding in perpetuity and, as
# such, survives the term of the Agreement.

# FEE: Provided Licensee abides completely by the terms and conditions of this
# Agreement, there is no fee due to Licensor for Licensee's use of the
# Software in accordance with this Agreement.

# DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
# OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
# FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
# BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
# RELATED MATERIALS.

# SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
# provided as part of this Agreement.  

# EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
# permitted under applicable law, Licensor shall not be liable for direct,
# indirect, special, incidental, or consequential damages or lost profits
# related to Licensee's use of and/or inability to use the Software, even if
# Licensor is advised of the possibility of such damage.

# EXPORT REGULATION: Licensee agrees to comply with any and all applicable
# export control laws, regulations, and/or other laws related to embargoes and
# sanction programs administered by law.

# SEVERABILITY: If any provision(s) of this Agreement shall be held to be
# invalid, illegal, or unenforceable by a court or other tribunal of competent
# jurisdiction, the validity, legality and enforceability of the remaining
# provisions shall not in any way be affected or impaired thereby.

# NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
# or remedy under this Agreement shall be construed as a waiver of any future
# or other exercise of such right or remedy by Licensor.

# GOVERNING LAW: This Agreement shall be construed and enforced in accordance
# with the laws of Germany without reference to conflict of laws principles.
# You consent to the personal jurisdiction of the courts of this country and
# waive their rights to venue outside of Germany.

# ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
# entire agreement between Licensee and Licensor as to the matter set forth
# herein and supersedes any previous agreements, understandings, and
# arrangements between the parties relating hereto.

#        THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.



import spacy
from tqdm import tqdm
from lfp_utils import run_in_parallel, code_switch_span, update_span, tup_match, tup_match_list
import json
import sys
import random
import argparse
from functools import partial

ja_parser = spacy.load("ja_core_news_sm", disable=["ner"])
de_parser = spacy.load("de_core_news_sm", disable=["ner"])
en_parser = spacy.load("en_core_web_sm", disable=["ner"])

ADP_IND = -1
ADP_POS = '_CASEMARKER'

def parsing(para_sents, lang='ja'):

    en_sent = para_sents[1]
    if lang == 'ja':
        tgt_sent = para_sents[0].strip('\n')
        tgt_tokens = [t.text.strip(' ') for t in ja_parser(tgt_sent)]
        # en_tokens = en_sent.strip('\n').split(' ')
        curr = f"{' '.join(tgt_tokens)} ||| {en_sent}"
    elif lang == 'de':
        de_sent = para_sents[0].strip('\n')
        en_sent = para_sents[1]
        curr = f"{de_sent} ||| {en_sent}"    
    return curr

def after_token(i, curr_en, reordered, ori_data, en_tokens, en2tar, used, absent):
    
    offset = 1
    if curr_en not in used:
        reordered["sentence"].append(en_tokens[curr_en])
        reordered["sentence2index"].append(ori_data["sentence2index"][curr_en])
        reordered["pos_tag"].append(ori_data["pos_tag"][curr_en])
        reordered["pos2index"].append(ori_data["pos2index"][curr_en])
        used.append(curr_en) 
                                
    while curr_en + offset in absent:
        tok = curr_en + offset
        reordered["sentence"].append(en_tokens[tok])
        reordered["sentence2index"].append(ori_data["sentence2index"][tok])
        reordered["pos_tag"].append(ori_data["pos_tag"][tok])
        reordered["pos2index"].append(ori_data["pos2index"][tok])      
        en2tar[tok] = i   
        absent.remove(tok)
        used.append(tok)
        offset += 1 
    
    return reordered, en2tar, used, absent

def after_span(i, curr_en, reordered, ori_data, en_tokens, en2tar, used, absent, buff, in_span):

    offset = 1
    if curr_en not in used:
        reordered["sentence"].append(en_tokens[curr_en])
        reordered["sentence2index"].append(ori_data["sentence2index"][curr_en])
        reordered["pos_tag"].append(ori_data["pos_tag"][curr_en])
        reordered["pos2index"].append(ori_data["pos2index"][curr_en])
        used.append(curr_en)

    while curr_en + offset in absent:
        tok = curr_en + offset
        buff.append(tok)
        absent.remove(tok)
        used.append(tok)
        offset += 1 

    if curr_en not in in_span: 
        for tok in buff:
            # print(f"{en_tokens[tok]} appended after {reordered['sentence']}")
            reordered["sentence"].insert(-1, en_tokens[tok])
            reordered["sentence2index"].insert(-1, ori_data["sentence2index"][tok])
            reordered["pos_tag"].insert(-1, ori_data["pos_tag"][tok])
            reordered["pos2index"].insert(-1, ori_data["pos2index"][tok])   
            en2tar[tok] = i 
        buff = []

    return reordered, en2tar, used, absent, buff
    

def after_token_en_based(reordered, ori_data, en_tokens, tgt_tokens, en2tar, tgt_deps = [], casemarker = False, casemarker_lst = []):

    tgt_toks = [-1 for i in tgt_tokens]
    new_ids = [-1 for i in tgt_tokens]
    
    for i in en2tar:
        if new_ids[en2tar[i]] != -1:
            new_ids.insert(en2tar[i] + 1, i)
            tgt_toks.insert(en2tar[i] + 1, tgt_tokens[en2tar[i]])
        else:
            new_ids[en2tar[i]] = i
            tgt_toks[en2tar[i]] = tgt_tokens[en2tar[i]]

    while -1 in new_ids:
        new_ids.remove(-1)

    for i in range(len(en_tokens)):
        if i not in en2tar:
            if i == 0:
                en2tar[i] = max(en2tar.values())
                new_ids.append(i)
            else:
                assert i-1 in en2tar
                en2tar[i] = en2tar[i-1]
                pos = new_ids.index(i-1) + 1
                new_ids.insert(pos, i)
        
    for tok in new_ids:
        reordered["sentence"].append(en_tokens[tok])
        reordered["sentence2index"].append(ori_data["sentence2index"][tok])
        reordered["pos_tag"].append(ori_data["pos_tag"][tok])
        reordered["pos2index"].append(ori_data["pos2index"][tok]) 
        
        if casemarker and tok in en2tar:
            tgt_id = en2tar[tok]
            if tgt_id + 1 < len(tgt_deps) and tgt_deps[tgt_id+1] == "case" and (tgt_tokens[tgt_id+1], tgt_id+1) not in casemarker_lst:
                    casemarker_lst.append((tgt_tokens[tgt_id+1], tgt_id+1))
                    reordered["sentence"].append(tgt_tokens[tgt_id+1])
                    reordered["sentence2index"].append(ADP_IND)
                    reordered["pos_tag"].append(ADP_POS)
                    reordered["pos2index"].append(ADP_IND)
                    ori_data["tuples"] = update_span(tok, tgt_tokens[tgt_id+1], ori_data["tuples"])                   

    return reordered, casemarker_lst

def run_lfp(para_sents, reorder = False, code_switch = False, casemarker = False,  percent = .2, lang = 'ja', policy = "token_en_based"):
    '''
    Code for enabling the LFP strategies.
    
    para_sents :: tuple of format (tgt_sent: str, en_sent: str, align: str, oie data: dict).
    reorder :: True/False for enabling/disabling reordering based on policy specified with `policy` argument.
    casemarker :: True/False for enabling/disabling casemarker insertion.
    code_switch :: True/False for enabling/disabling code-switching, with certain percent of aligned words with a sentence replaced by its counterpart in target language.
    percent :: Specifies the percentage of words that are code-switched to the target language.
    policy :: Specifies the reordering strategy. Three varients available: "after_token", "after_span", "after_token_en_based". The major one used in the report is "after_token_en_based".
    
    '''
    
    tgt_sent = para_sents[0].strip('\n')
    en_sent = para_sents[1]
    align = para_sents[2] 
    ori_data = para_sents[3]

    if lang == 'ja':
        tgt_tokens_ = [t.text.strip(' ') for t in ja_parser(tgt_sent)]
        tgt_deps_ = [t.dep_ for t in ja_parser(tgt_sent)] 
        tgt_tokens = []
        tgt_deps = []
        # deal with some tokenize issue with english tokens in japanese sentences 
        for i, token in enumerate(tgt_tokens_): 
            if ' ' in token:
                tgt_tokens.extend(token.split(' '))
                tgt_deps.extend([tgt_deps_[i] for j in token.split(' ')])
            else:
                tgt_tokens.append(token)
                tgt_deps.append(tgt_deps_[i])

    elif lang == 'de' or lang == 'ar':
        tgt_tokens = tgt_sent.replace('\xa0', ' ').strip('\n').split(' ') 
        tgt_deps = [t.dep_ for t in de_parser(tgt_sent)] 

    en_tokens = en_sent.replace('\xa0', ' ').strip('\n').split(' ') 

    tar2en = {}
    for a in align.split(' '):
        ja, en = a.strip('\n').split('-')
        if int(ja) in tar2en:
            continue
        tar2en[int(ja)] = int(en)

    tar2en = {k: v for k, v in sorted(tar2en.items(), key=lambda item: item[0])}
    en2tar = {v: k for k, v in sorted(tar2en.items(), key=lambda item: item[1])}
    absent = [i for i in range(len(en_tokens)) if i not in en2tar]
    used = []
    buff = []
    in_span = []
    casemarker_lst = []

    if code_switch == True:
        en_tokens_switched = {}
        for i in range(len(en_tokens)):
            if i in en2tar and tgt_tokens[en2tar[i]] != '':
                en_tokens_switched[i] = tgt_tokens[en2tar[i]]
        # randomly select aligned tokens to be code-switched based on percent.
        to_switch = random.sample(list(en_tokens_switched), k=int(len(en_tokens_switched) * percent))
        for i in to_switch:
            en_tokens[i] = en_tokens_switched[i]
            if tgt_deps[en2tar[i]] == "case" and casemarker:
                casemarker_lst.append((tgt_tokens[en2tar[i]], en2tar[i]))

        ori_data["tuples"] = code_switch_span(en_tokens, ori_data["tuples"])

    for tup in ori_data["tuples"]:
        in_span.extend(range(tup["rel_pos"][0], tup["rel_pos"][1]))
        in_span.extend(range(tup["arg0_pos"][0], tup["arg0_pos"][1]))
        for arg_span in tup["args_pos"]:
            in_span.extend(range(arg_span[0], arg_span[1]))
    
    reordered = {"sentence": [],
                "sentence2index": [], 
                "pos_tag": [], 
                "pos2index" : [],
                "tuples" : []}

    offset = len(casemarker_lst)
    
    if reorder:
        
        if policy == "token_en_based":
            reordered, casemarker_lst = after_token_en_based(reordered, ori_data, en_tokens, tgt_tokens, en2tar, 
                                                             tgt_deps, casemarker, casemarker_lst) 

        else:  
            for idx, i in enumerate(tar2en):
                
                curr_en = tar2en[i]
                
                if policy == "token":
                    reordered, en2tar, used, absent = after_token(i, curr_en, reordered, ori_data, 
                                                    en_tokens, en2tar, used, absent)
                    
                elif policy == "span":
                    reordered, en2tar, used, absent, buff = after_span(i, curr_en, reordered, ori_data, 
                                                    en_tokens, en2tar, used, absent, buff, in_span)           

                if casemarker:              
                    if i+1 < len(tgt_tokens) and tgt_deps[i+1] == "case" and (tgt_tokens[i+1], i+1) not in casemarker_lst:
                        casemarker_lst.append((tgt_tokens[i+1], i+1))
                        reordered["sentence"].append(tgt_tokens[i+1])
                        reordered["sentence2index"].append(ADP_IND)
                        reordered["pos_tag"].append(ADP_POS)
                        reordered["pos2index"].append(ADP_IND)
                        ori_data["tuples"] = update_span(curr_en, tgt_tokens[i+1], ori_data["tuples"]) 
         
            while buff != []:
                tok = buff[0]
                reordered["sentence"].append(en_tokens[tok])
                reordered["sentence2index"].append(ori_data["sentence2index"][tok])
                reordered["pos_tag"].append(ori_data["pos_tag"][tok])
                reordered["pos2index"].append(ori_data["pos2index"][tok])
                en2tar[tok] = max(en2tar.values())
                buff.remove(tok)
                
            while absent != []:
                tok = absent[-1]
                reordered["sentence"].append(en_tokens[tok])
                reordered["sentence2index"].append(ori_data["sentence2index"][tok])
                reordered["pos_tag"].append(ori_data["pos_tag"][tok])
                reordered["pos2index"].append(ori_data["pos2index"][tok])
                en2tar[tok] = max(en2tar.values()) 
                absent.remove(tok)   

        assert len(en_tokens) + len(casemarker_lst) - offset == len(reordered["sentence"])
    
    else: # no reordering
        for i in range(len(en_tokens)):
        
            reordered["sentence"].append(en_tokens[i])
            reordered["sentence2index"].append(ori_data["sentence2index"][i])
            reordered["pos_tag"].append(ori_data["pos_tag"][i])
            reordered["pos2index"].append(ori_data["pos2index"][i])

            if casemarker and i in en2tar:
                pos = en2tar[i]
                if pos+1 < len(tgt_tokens) and tgt_deps[pos + 1] == "case" and (tgt_tokens[pos+1], pos+1) not in casemarker_lst:
                    reordered["sentence"].append(tgt_tokens[pos+1])
                    reordered["sentence2index"].append(ADP_IND)
                    reordered["pos_tag"].append(ADP_POS)
                    reordered["pos2index"].append(ADP_IND) 
                    ori_data["tuples"] = update_span(i, tgt_tokens[pos+1], ori_data["tuples"]) 
        en2tar = {i:i for i in range(len(en_tokens))}
    # map tuples.
    # print(ori_data["tuples"])
    for tup in ori_data["tuples"]:
        new_tup = {"score": tup["score"], "context": tup["context"]}
        new_tup["arg0"], new_tup["arg0_pos"] = tup_match(tup["arg0"], tup["arg0_pos"], en2tar)
        new_tup["relation"], new_tup["rel_pos"] = tup_match(tup["relation"], tup["rel_pos"], en2tar)
        new_tup["args"], new_tup["args_pos"] = tup_match_list(tup["args"], tup["args_pos"], en2tar)
        reordered["tuples"].append(new_tup)


    reordered["sentence"] = ' '.join(reordered["sentence"])


    return reordered

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--lang', type=str, choices = ['de', 'ar', 'ja'], help='target language', required=True)
    parser.add_argument('--name', type=str, help='identifier of data to be generated', required=True)
    
    parser.add_argument('--ro', action='store_true', help='linguistic feature projection with word reordering')
    parser.add_argument('--cs', action='store_true', help='linguistic feature projection with code switching')
    parser.add_argument('--cm', action='store_true', help='linguistic feature projection with case marker insertion')
    
    parser.add_argument('--output_dir', type=str, help='path to write the outputs', default='')
    parser.add_argument('--base_data', type=str, help='path to training data', default='/home/youmima/jpoie/mnt_data/data/structured_data.json')
    parser.add_argument('--src_sents', type=str, help='path to training data (sentences only)', default='/home/youmima/jpoie/preprocess/train_sents.en')
    parser.add_argument('--tgt_sents', type=str, help='path to translated sentences in the target language', default='/home/youmima/jpoie/preprocess/train_sents.ja')
    
    parser.add_argument('--ignore', type=str,help='path to dictionary containing ids of sentence pairs to be ingored', default=None)
    parser.add_argument('--align', type=str, help='path to word alignments between original sentences and translated sentences', default=None)
    
    args = parser.parse_args()
    
    ignore_ids = []
    if args.ignore != None:
        with open(args.ignore, "r") as f:
            ignore_ids = json.load(f)
    
    with open(args.tgt_sents, "r") as tarf, open(args.src_sents, "r") as srcf:
        tgt_sents = [sent for i, sent in enumerate(tarf.readlines()) if i not in ignore_ids]
        src_sents = [sent for i, sent in enumerate(srcf.readlines()) if i not in ignore_ids]
        assert len(tgt_sents) == len(src_sents)
    
    
    if args.align == None:
        para_sents = list(zip(tgt_sents, src_sents))
        aligns = run_in_parallel(parsing, para_sents, 10)
          
        with open(f"{args.output_dir}/train_{args.lang}2en.align", "w") as wf:
            for align in tqdm(aligns, desc="writing file for alignments"):
                wf.write(align)
        print(f"Data to be aligned written into: {args.output_dir}/train_{args.lang}2en.align")

        exit(0)
        
    else:
        with open(args.align, "r") as f:
            aligns = f.readlines()   

    with open(args.base_data, "r") as f:
        ori_data = [data for i,data in enumerate(json.load(f)) if i not in ignore_ids]

    
    assert len(aligns) == len(tgt_sents)
    assert len(aligns) == len(ori_data)
    
    para_sents = list(zip(tgt_sents, src_sents, aligns, ori_data))

    print("Base data and word alignment loaded.")
    print("=" * 20 + "Lingusitic Feature Projection" + "=" * 20)
    print(f"Word Reordering: {args.ro}")
    print(f"Code Switching: {args.cs}")
    print(f"Case Marker Insertion: {args.cm}")

    run_lfp_ = partial(run_lfp, reorder = args.ro, code_switch = args.cs, casemarker = args.cm, lang = args.lang)

    reordered_dict = run_in_parallel(run_lfp_, para_sents, 10)

    with open(f"{args.output_dir}/structured_data_{args.name}.json", "w") as wf:
        json.dump(reordered_dict, wf)
    
    print(f"Data after LFP stored into {args.output_dir}/structured_data_{args.name}.json.")