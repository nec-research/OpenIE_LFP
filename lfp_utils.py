#        Improving Cross-Lingual Transfer for Open Information Extraction with Linguistic Feature Projection 
	  
#   File:     lfp_utils.py
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



from tqdm import tqdm
from multiprocessing import Pool

def run_in_parallel(func, records, processes, desc="", leave=True, total=None):
    """
    processes: number of processes
    records: inputs in tuples
    func: function for multi-processing
    desc: name on process bar
    """
    pool = Pool(processes)
    imap = pool.imap(func, records, chunksize=1)
    if total is None:
        total=len(records)
    result = list(tqdm(imap, ascii=True, desc=desc, leave=leave, total=total))
    pool.close()
    pool.terminate()
    pool.join()
    del pool
    return result 

def code_switch_span(en_tokens_switched, tuples):
    
    ''' Code switching of tuples based on switched english tokens. ''' 
    
    new_tuples = []
    
    for tup in tuples:
        new_tup = {"score": tup["score"], "context": tup["context"], "rel_pos": tup["rel_pos"], "arg0_pos": tup["arg0_pos"], "args_pos": tup["args_pos"]}
        rel_span = range(tup["rel_pos"][0], tup["rel_pos"][1] + 1)
        rel_toks = [en_tokens_switched[i] for i in rel_span]
        arg0_span = range(tup["arg0_pos"][0], tup["arg0_pos"][1] + 1)
        arg0_toks = [en_tokens_switched[i] for i in arg0_span]
        args_span = [range(arg[0], arg[1]+1) for arg in tup["args_pos"]]
        args_toks = [[en_tokens_switched[i] for i in span] for span in args_span]
        new_tup["relation"] = ' '.join(rel_toks)
        new_tup["arg0"] = ' '.join(arg0_toks)
        new_tup["args"] = [' '.join(arg) for arg in args_toks]
        new_tuples.append(new_tup)

    return new_tuples


def update_span(en_idx, casemarker, tuples):

    ''' update the casemarker info in tuples. 
    Basically, the code inserts the casemarker (given as argument) after the token corresponding to en_idx.
    '''
    
    new_tuples = []
    
    for tup in tuples:
        new_tup = {"score": tup["score"], "context": tup["context"], "rel_pos": tup["rel_pos"], "arg0_pos": tup["arg0_pos"], "args_pos": tup["args_pos"]}
        rel_span = range(tup["rel_pos"][0], tup["rel_pos"][1] + 1)
        rel_toks = tup["relation"].split()
        arg0_span = range(tup["arg0_pos"][0], tup["arg0_pos"][1] + 1)
        arg0_toks = tup["arg0"].split()
        args_span = [range(arg[0], arg[1]+1) for arg in tup["args_pos"]]
        args_toks = [arg.split() for arg in tup["args"]]
        if en_idx in rel_span:
            rel_toks[rel_span.index(en_idx)] += f"<sep>{casemarker}"
        elif en_idx in arg0_span:
            arg0_toks[arg0_span.index(en_idx)] += f"<sep>{casemarker}"
        else:
            for i, span in enumerate(args_span):
                if en_idx in span:
                    args_toks[i][span.index(en_idx)] += f"<sep>{casemarker}"
                    break
        new_tup["relation"] = ' '.join(rel_toks)
        new_tup["arg0"] = ' '.join(arg0_toks)
        new_tup["args"] = [' '.join(arg) for arg in args_toks]
        new_tuples.append(new_tup)

    return new_tuples


def tup_match(old_name, old_pos, en2tar):
   
    ''' matching tuples with (name, pos) = (old name, old pos) into new tuples.'''
     
    start, end = old_pos[0], old_pos[1]
    src_seq = [j for j in range(start, end+1)]
    tar_seq = []
    old_name_splited = old_name.split(' ')
    while '' in old_name_splited:
        old_name_splited.remove('')
    new_name = []

    for i, j in enumerate(src_seq):
        if j in en2tar:
            tar_seq.append(en2tar[j])
            new_name.append((old_name_splited[i], en2tar[j]))
    
    new_name = sorted(new_name, key=lambda x: x[1])

    tar_name = ' '.join([n[0].replace('<sep>', ' ') for n in new_name])
    tar_seq.sort()

    return tar_name, tar_seq

def tup_match_list(old_names, old_poss, en2tar):

    ''' call tup_match() for each element in a list. 
    Basically used for <args> key-value pair in original data.'''
    
    assert len(old_names) == len(old_poss)
    names = []
    locs = []
    for i, old_name in enumerate(old_names):
        name, loc = tup_match(old_name, old_poss[i], en2tar)
        names.append(name)
        locs.append(loc)
    return names, locs
