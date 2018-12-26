# encoding:utf-8

from collections import OrderedDict
import pandas as pd
import re
from copy import deepcopy

from newNER.abstract import abstract
from modules.ner_rule.ner_rule import ner_rule

class productNER(abstract):
    def __init__(self,txt):
        self.txt = txt
    
    def recognize(self,*ner_list):
        if len(ner_list) != 0:
            checks = [ self.txt == ner["txt"] for ner in ner_list]
            if False in checks:
                print("txt not match")
                return None
        result = []
        self.dfs(*ner_list,idx=0,result=result)
        candidates = []
        if len(result) != 0:
            for res in result:
                cand = self.recognizer(*res)
                candidates += cand
        else:
            candidates = self.recognizer()
        result = OrderedDict()
        result["txt"] = self.txt
        result["candidates"] = candidates
        return result

    def dfs(self,*ner_list,idx=0,result=None):
        '''
        ner_list: json object
        ner_list: [
            {   "txt": ...,   
                "candidates":[
                    {
                        "realStart":xxx,
                        "realEnd":yyy,
                        "entity":["",""],
                        "name":"time",
                        "sn":"xxx"
                    }
                ]
            },
            {}
        ]
        '''
        if idx >= len(ner_list) or idx < 0:
            return
        ner = ner_list[idx]
        conflicts = [set() for _ in range(len(self.txt))]
        for i in range(len(ner["candidates"])):
            candidate = ner["candidates"][i]
            for j in range(int(candidate["realStart"]),int(candidate["realEnd"])):
                conflicts[j].add(i)
        conflicts = list(set([ "_{}_".format("_(\d+_)*".join(map(str,sorted(list(conflict))))) for conflict in conflicts if len(conflict) > 1 ]))
        conflicts = [ re.compile(conflict) for conflict in conflicts ]

        num_candidates = len(ner["candidates"])
        all_idx_combinations = []
        for r in range(1,num_candidates+1):
            tmp = []
            self.combinations(ner["candidates"],0,r,all_idx_combinations,tmp)
        
        isConflictToHistory = []
        for i in range(num_candidates):
            flag = False
            for res in result:
                for other_ner in res:
                    if not (int(ner["candidates"][i]["realStart"]) >= int(other_ner["realEnd"]) or int(ner["candidates"][i]["realEnd"]) <= int(other_ner["realStart"])):
                        flag = True
                        break
                if flag:
                    break
            isConflictToHistory.append(flag)
        is_empty = True
        if result is not None and len(result) != 0:
            is_empty = False
        copy_result = deepcopy(result)
        len_result = len(result)
        cursor = 0
        while cursor < len_result:
            result.pop(0)
            cursor += 1
        for combinations in all_idx_combinations:
            if sum([ isConflictToHistory[combination] for combination in combinations]) > 0:
                continue
            if len(combinations) > 1:
                flag = False
                target = "_{}_".format("_".join(map(str,combinations)))
                for conflict in conflicts:
                    if conflict.search(target) is not None:
                        flag = True
                        break
                if flag:
                    continue            
            if not is_empty:
                tmp_result = deepcopy(copy_result)
                for res in tmp_result:
                    res += [ner["candidates"][combination] for combination in combinations]
                result += tmp_result
            else:
                result.append([ner["candidates"][combination] for combination in combinations])
        self.dfs(*ner_list,idx=idx+1,result=result)

    def recognizer(self,*other_ners):
        obj_ner_rule = ner_rule(*other_ners,txt=self.txt)
        return obj_ner_rule.recognizer()
    
    def combinations(self,all_elements,idx,r,result,tmp):
        if r == 0:
            result.append(tmp)
            return
        if r < 0 or r > len(all_elements) or idx >= len(all_elements) or idx < 0:
            return
        self.combinations(all_elements,idx+1,r-1,result,tmp=tmp+[idx])
        self.combinations(all_elements,idx+1,r,result,tmp=tmp)

if __name__ == "__main__":
    import json
    txt = "钛慷金满仓B3年交生存金怎么返"
    ner_json = {
                "txt":"钛慷金满仓B3年交生存金怎么返",
                "candidates":[
                    {
                        "realStart":"6",
                        "realEnd":"9",
                        "standard":"3年交",
                        "name":"time",
                        "sn":"time-reg"
                    }
                ]}
    ner_json1 = {
                "txt":"钛慷金满仓B3年交生存金怎么返",
                "candidates":[
                    {
                        "realStart":"0",
                        "realEnd":"2",
                        "standard":"泰康",
                        "name":"time",
                        "sn":"time-reg"
                    },
                    {
                        "realStart":"1",
                        "realEnd":"2",
                        "standard":"泰康",
                        "name":"time",
                        "sn":"time-reg"
                    },
                    {
                        "realStart":"0",
                        "realEnd":"1",
                        "standard":"泰康",
                        "name":"time",
                        "sn":"time-reg"
                    }
                ]}
    p = productNER(txt)
    print(json.dumps(p.recognize(ner_json,ner_json1),ensure_ascii=False))


    # df = pd.read_excel("input/标注问题_taikang_952_760entity_corpus.xlsx")
    # queries = []
    # entities = []
    # standards = []
    # segs = []
    # for i in range(len(df)):
    #     query = str(df.loc[i][0]).strip().replace(" ","")
    #     queries.append(query)
    #     p = productNER(query)
    #     recog = p.recognize()
    #     entity = []
    #     standard = []
    #     segment = []
    #     for cand in recog["candidates"]:
    #         entity.append(cand["entity"])
    #         standard.append("{}:{}".format(cand["entity"],cand["standard"]))
    #         segment.append(cand["segments"])
    #     entities.append("/{}/".format("/".join(entity)))
    #     standards.append("\n".join(standard))
    #     segs.append("\n".join(segment))
    # result = OrderedDict()
    # result["query"] = queries
    # result["segs"] = segs
    # result["entities"] = entities
    # result["standard"] = standards
    # df = pd.DataFrame(result)
    # df.to_excel("output/result.xlsx",index=False)

    