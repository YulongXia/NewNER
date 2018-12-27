# encoding: utf-8
import jieba
import os
import re
from collections import OrderedDict
from functools import reduce
import marisa_trie

jieba.load_userdict(os.path.join(os.path.dirname(os.path.abspath(__file__)),"resources/userdict/userdict.txt"))

class ner_rule(object):
    def __init__(self,*other_ners,txt=""):
        self.other_ners = other_ners
        self.txt = txt
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"resources/mapping/mapping.txt"),"r",encoding="utf-8") as f:
            self.mapping = OrderedDict()
            for line in f.readlines():
                if len(line.strip()) == 0 or line.strip().startswith("#"):
                    continue
                parts = line.split()
                self.mapping[parts[0]] = parts[1].split(",")
        self.resources = self.getResources()
        self.extras = self.other_ners[:]
        self.extras = sorted(self.extras,key=lambda x:int(x["realStart"]))
        for other_ner in self.other_ners:
            word = self.txt[int(other_ner["realStart"]):int(other_ner["realEnd"])]
            for word_set in self.mapping[other_ner["name"]]:
                self.resources[word_set].append(word)

    def cut(self,txt):
        info = []
        left = 0
        for extra in self.extras:
            if left != int(extra["realStart"]):
                # 1: 需要分词 
                # 0: 不需要分词
                info.append([txt[left:int(extra["realStart"])],1])
                info.append([txt[int(extra["realStart"]):int(extra["realEnd"])],0])
            else:
                info.append([txt[int(extra["realStart"]):int(extra["realEnd"])],0])
            left = int(extra["realEnd"])
        if left < len(txt):
            info.append([txt[left:len(txt)],1])
        segments = []
        for section,flag in info:
            if flag == 1:
                segments += list(jieba.cut(section))
            else:
                segments.append(section)
        return segments
        

    def recognizer(self):
        segments = self.cut(self.txt)
        self.adjust_seg(segments)
        start = 0 
        end = len(segments)
        tmp = []
        entity_link_info = []
        guess = []
        heads,tails,flag,kernel_idx = self.search(segments,start,end)
        guess.append([heads,tails,flag,kernel_idx])
        self.post_processing(segments=segments,heads=heads,tails=tails,flag=flag,kernel_idx=kernel_idx,ambiguous_suffix=self.resources["ambiguous_suffix"],tail_plus_one_patterns=self.resources["tail_plus_one_patterns"],tmp=tmp,entity_link_info=entity_link_info,invalid_single_word_as_entity=self.resources["invalid_single_word_as_entity"],skips=self.resources["skips"],tail_minus_one_and_tail_stop_patterns=self.resources["tail_minus_one_and_tail_stop_patterns"],        tail_minus_one_and_tail_cut_tail_patterns=self.resources["tail_minus_one_and_tail_cut_tail_patterns"],ambiguous_prefix=self.resources["ambiguous_prefix"])
        while len(tails) > 0 and max(tails) < end:
            start = max(tails) + 1
            heads,tails,flag,kernel_idx = self.search(segments,start,end)
            guess.append([heads,tails,flag,kernel_idx])
            self.post_processing(segments=segments,heads=heads,tails=tails,flag=flag,kernel_idx=kernel_idx,ambiguous_suffix=self.resources["ambiguous_suffix"],tail_plus_one_patterns=self.resources["tail_plus_one_patterns"],tmp=tmp,entity_link_info=entity_link_info,invalid_single_word_as_entity=self.resources["invalid_single_word_as_entity"],skips=self.resources["skips"],tail_minus_one_and_tail_stop_patterns=self.resources["tail_minus_one_and_tail_stop_patterns"],        tail_minus_one_and_tail_cut_tail_patterns=self.resources["tail_minus_one_and_tail_cut_tail_patterns"],ambiguous_prefix=self.resources["ambiguous_prefix"])    
        if len(guess) > 1:
            # sorted by kernel_idx
            guess = sorted(guess,key=lambda x:x[3] if x[2] else 99999999 )
            # sorted by flag in descend order e.g.[True,True,False]
            guess = sorted(guess,key=lambda x:x[2],reverse=True)
            # /19/20/
            all_string = "/" + "/".join([str(i) for i in range(len(segments))]) + "/"
            part = "/"
            idx_guess = 0
            while guess[idx_guess][2] == True:
                part += str(guess[idx_guess][3]) + "/"
                idx_guess += 1
            if idx_guess >= 2 and part in all_string:
                heads = guess[0][0]
                tails = guess[idx_guess - 1][1]
                self.post_processing(segments=segments,heads=heads,tails=tails,flag="guess",kernel_idx=[guess[0][3],guess[idx_guess - 1][3]],ambiguous_suffix=self.resources["ambiguous_suffix"],tail_plus_one_patterns=self.resources["tail_plus_one_patterns"],tmp=tmp,entity_link_info=entity_link_info,invalid_single_word_as_entity=self.resources["invalid_single_word_as_entity"],skips=self.resources["skips"],tail_minus_one_and_tail_stop_patterns=self.resources["tail_minus_one_and_tail_stop_patterns"],        tail_minus_one_and_tail_cut_tail_patterns=self.resources["tail_minus_one_and_tail_cut_tail_patterns"],ambiguous_prefix=self.resources["ambiguous_prefix"])
        candidates = OrderedDict()
        for one in entity_link_info:
            for entity,info in one.items():
                if str(info["realStart"]) + "-" + str(info["realEnd"]) in candidates:
                    continue
                cur_set = self.resources["standard_entity"]
                kernel_word = info["kernel_word"]
                if len(kernel_word) != 0:
                    cur_filter = kernel_word
                    tmp = []
                    for word in cur_set:
                        if word.find(cur_filter) != -1 :
                            tmp.append(word)
                    cur_set = tmp
                entity_segments = info["segments"]
                for segment in entity_segments:
                    if segment in self.resources["prefix"] or segment in self.resources["suffix"] or segment in self.resources["skips"] or segment == kernel_word:
                        continue
                    tmp = []
                    cur_filter = segment
                    for word in cur_set:
                        if word.find(cur_filter) != -1:
                            tmp.append(word)
                    cur_set = tmp
                candidate = OrderedDict()
                candidate["realStart"] = str(info["realStart"])
                candidate["realEnd"] = str(info["realEnd"])
                candidate["entity"] = entity
                candidate["standard"] = "Not found"
                if len(cur_set) != 0:
                    candidate["standard"] = "/".join(cur_set)
                candidate["name"] = "product"
                candidate["sn"] = "productNER-ner_rule"
                candidate["segments"] = "/".join(segments)
                key = candidate["realStart"] + '-' + candidate["realEnd"]
                candidates[key] = candidate
        return list(candidates.values())
    
    def post_processing(self,**kwargs):
        segments = kwargs["segments"]
        heads = kwargs["heads"]
        tails = kwargs["tails"]
        flag = kwargs["flag"]
        kernel_idx = kwargs["kernel_idx"]
        ambiguous_suffix = kwargs["ambiguous_suffix"]
        tail_plus_one_patterns = kwargs["tail_plus_one_patterns"]
        tmp = kwargs["tmp"]
        entity_link_info = kwargs["entity_link_info"]
        invalid_single_word_as_entity = kwargs["invalid_single_word_as_entity"]
        skips = kwargs["skips"]
        ambiguous_prefix = kwargs["ambiguous_prefix"]
        tail_minus_one_and_tail_stop_patterns = kwargs["tail_minus_one_and_tail_stop_patterns"]
        tail_minus_one_and_tail_cut_tail_patterns = kwargs["tail_minus_one_and_tail_cut_tail_patterns"]
        kernel_word = ""
        if flag == True:
            kernel_word = segments[kernel_idx]
        if flag == "guess":
            kernel_word = "".join(segments[kernel_idx[0]:kernel_idx[1]+1])

        # print("/".join(segments))
        one = dict()
        while len(heads) != 0:
            head = heads.pop(0)
            for tail in tails:
                if tail < head:
                    continue
                pairs = []
                ending_label = False
                while not ending_label:
                    n = 0
                    if  tail - head >= 1 and segments[head + 1] in segments[head]:
                        head = head + 1
                        n += 1
                    if tail - head >= 1 and segments[tail] in segments[tail-1]:
                        tail = tail - 1
                        n += 1
                    if tail - head >= 1 and (segments[head] in ambiguous_prefix or segments[head] in skips) and head + 1 not in heads:
                        heads.append(head + 1)
                        n += 1
                    if tail - head >= 1 and (segments[tail] in ambiguous_suffix or segments[tail] in skips):
                        pairs.append([head,tail])
                        tail = tail - 1
                        n += 1
                    # e康豁免保险费疾病重大疾病豁免保险费责任
                    if tail - head >= 2 and sum([re.compile(pattern).match("".join(segments[tail-1:tail+1])) is not None for pattern in tail_minus_one_and_tail_stop_patterns]) >= 1 :
                        tail = tail - 2
                        n += 1
                    if tail - head >= 2 and sum([re.compile(pattern).match("".join(segments[tail-1:tail+1])) is not None for pattern in tail_minus_one_and_tail_cut_tail_patterns]) >= 1:
                        tail = tail - 1
                        n += 1
                    s = "".join(segments[tail:tail+2])
                    if len([True for pattern in tail_plus_one_patterns if pattern.match(s) is not None]) != 0:
                        tail = tail - 1
                        n += 1
                    if n == 0:
                        ending_label = True
                pairs.append([head,tail])
                for head,tail in pairs:
                    real_head = head
                    real_tail = tail
                    for it in range(head,tail+1):
                        if re.compile(r"^[,，.。?？!！、]$").match(segments[it]) is not None:
                            if flag:
                                if it > kernel_idx and it <= real_tail:
                                    real_tail = it - 1
                                elif it < kernel_idx and it >= real_head:
                                    real_head = it + 1
                            else:
                                if it <= real_tail:
                                    real_tail = it - 1
                    if real_tail == real_head and segments[real_head] in invalid_single_word_as_entity:
                        continue
                    entity = "".join(segments[real_head:real_tail+1])
                    if real_tail >= real_head:
                        real_head,real_tail = self.remove_invalid_brackets(segments,real_head,real_tail)
                        entity = "".join(segments[real_head:real_tail+1])
                        if entity not in one:
                            accumulate = []
                            tmp_accumulate = 0
                            for _idx in range(len(segments)):
                                tmp_accumulate += len(segments[_idx])
                                accumulate.append(tmp_accumulate)
                            one[entity] = {"kernel_word":kernel_word,"segments":segments[real_head:real_tail+1],"realStart":accumulate[real_head]-len(segments[real_head]),"realEnd":accumulate[real_tail]}
        if len(one) != 0:
            tmp.append("^".join(list(one.keys())))
            entity_link_info.append(one)

    def remove_invalid_brackets(self,segments,real_head,real_tail):
        match = {
            "(": {"direct":"left","value":1},
            ")": {"direct":"right","value":1},
            "（": {"direct":"left","value":1},
            "）": {"direct":"right","value":1},
            "[": {"direct":"left","value":2},
            "]": {"direct":"right","value":2},
            "【": {"direct":"left","value":2},
            "】": {"direct":"right","value":2},
            "{": {"direct":"left","value":3},
            "}": {"direct":"right","value":3},
            "『": {"direct":"left","value":3},
            "』": {"direct":"right","value":3},
            "<": {"direct":"left","value":4},
            ">": {"direct":"right","value":4},
            "《": {"direct":"left","value":4},
            "》": {"direct":"right","value":4},
            
        }
        drops = [1]
        while len(drops) != 0:
            # processing invalid brackets
            stack = []
            drops = [] 
            for i in range(real_head,real_tail+1):
                character = segments[i]
                if character in match:
                    # e.g. node[0]: 3, node[1]: "【" , node[2]: {"direct":"right","value":3}
                    node = [i,character,match[character]]
                    if node[2]["direct"] == "left":
                        stack.insert(0,node)
                    if node[2]["direct"] == "right":
                        if len(stack) != 0:
                            pop_node = stack.pop(0)
                            if pop_node[2]["value"] != node[2]["value"]:
                                drops.append(pop_node)
                                drops.append(node)
                        else:
                            drops.append(node)
            while len(stack) != 0:
                drops.append(stack.pop(0))
            
            if len(drops) != 0:
                last_tail = None
                drops = sorted(drops,key=lambda x:x[0],reverse=True)
                if real_tail == drops[0][0]:
                    last_tail = drops[0]
                    for i in range(1,len(drops)):
                        if last_tail[0] - drops[i][0] == 1:
                            last_tail = drops[i]
                        else:
                            break
                if last_tail is not None:
                    real_tail = last_tail[0] - 1

                last_head = None
                drops = sorted(drops,key=lambda x:x[0],reverse=False)
                if real_head == drops[0][0]:
                    last_head = drops[0]
                    for i in range(1,len(drops)):
                        if drops[i][0] - last_head[0] == 1:
                            last_head = drops[i]
                        else:
                            break
                if last_head is not None:
                    real_head = last_head[0] + 1

            # processing valid brackets but which are both on the both-end
            stack = []
            drops = [] 
            for i in range(real_head,real_tail+1):
                character = segments[i]
                if character in match:
                    # e.g. node[0]: 3, node[1]: "【" , node[2]: {"direct":"right","value":3}
                    node = [i,character,match[character]]
                    if node[2]["direct"] == "left":
                        stack.insert(0,node)
                    if node[2]["direct"] == "right":
                        if len(stack) != 0:
                            pop_node = stack.pop(0)
                            if (abs(node[0]-pop_node[0]) == 1 and (node[0] == real_tail or pop_node[0] == real_head)) or (node[2]["value"] == pop_node[2]["value"] and node[0] == real_tail and pop_node[0] == real_head):
                                drops.append(pop_node)
                                drops.append(node)
                if len(drops) != 0:
                    last_tail = None
                    drops = sorted(drops,key=lambda x:x[0],reverse=False)
                    if real_tail == drops[-1][0]:
                        last_tail = drops[-1]
                        for i in range(len(drops) - 2,-1,-1):
                            if last_tail[0] - drops[i][0] == 1:
                                last_tail = drops[i]
                            else:
                                break
                    if last_tail is not None:
                        real_tail = last_tail[0] - 1

                    last_head = None
                    if real_head == drops[0][0]:
                        last_head = drops[0]
                        for i in range(1,len(drops)):
                            if drops[i][0] - last_head[0] == 1:
                                last_head = drops[i]
                            else:
                                break
                    if last_head is not None:
                        real_head = last_head[0] + 1
                    

        return real_head,real_tail


    def search(self,segments,start,end):
        forward = self.resources["forward"]
        backward = self.resources["backward"]
        whole_without_kernel = self.resources["whole_without_kernel"]
        skips = self.resources["skips"]
        kernel = self.resources["kernel"]
        regex_limited = self.resources["regex_limited"]
        mytrie = self.resources["mytrie"]
        properties = self.resources["properties"]
        if start >= end or start < 0:
            return [],[],False,0
        flag = False
        for i in range(start,end):
            seg = segments[i]
            if seg in kernel:
                flag = True
                idx = i
                break
        heads = []
        tails = []
        if flag:
            self.forwardsearch(segments,idx+1,forward,tails,start,end,skips,regex_limited,mytrie,properties)
            self.backwardsearch(segments,idx-1,backward,heads,start,end,skips,regex_limited,properties)  
        else:
            idx = None
            for i in range(start,end):
                seg = segments[i]
                if seg in whole_without_kernel:
                    idx = i
                    break
            if idx is not None:
                heads.append(idx)
                self.forwardsearch(segments,idx+1,forward,tails,start,end,skips,regex_limited,mytrie,properties)        
        return heads,tails,flag,idx

    def backwardsearch(self,segments,i,backward,heads,start,end,skips,regex_limited,properties):
        if i < start :
            heads.append(start)
            return
        if segments[i] in backward or segments[i] in skips or len([ True for pattern in regex_limited if pattern.match(segments[i]) is not None ] ) > 0 :
            self.backwardsearch(segments,i-1,backward,heads,start,end,skips,regex_limited,properties)
        else:
            heads.append(i+1)
            if "".join(segments[i:i+2]) in properties and i < end - 2:
                heads.append(i+2)
        return




    def forwardsearch(self,segments,i,forward,tails,start,end,skips,regex_limited,mytrie,properties):
        if i >= end :
            tails.append(i - 1)
            return
        if segments[i] in forward or segments[i] in skips or len([ True for pattern in regex_limited if pattern.match(segments[i]) is not None ] ) > 0:
            self.forwardsearch(segments,i+1,forward,tails,start,end,skips,regex_limited,mytrie,properties)
        else:
            tails.append(i-1)
            if len(segments[i]) > 1:
                prefix_list = mytrie.prefixes(segments[i])
                if len(prefix_list) > 0:
                    longest_prefix = reduce(self.compare,prefix_list)
                    split_a = segments[i][0:len(longest_prefix)]
                    split_b = segments[i][len(longest_prefix):]
                    segments[i] = split_a
                    segments.insert(i+1,split_b)
                    tails[-1] = i
                    i += 1
            if "".join(segments[i-1:i+1]) in properties and i > start + 1:
                tails.append(i-2)
                
        return

    def compare(self,s1,s2):
        s1 = str(s1)
        s2 = str(s2)
        if len(s1) >= len(s2):
            return s1
        else:
            return s2

    def adjust_seg(self,segments,idx=0,start=0,flag=False):
        if idx >= len(segments):
            if flag:
                if idx - start > 1:
                    word = []
                    for _ in range(start,idx):
                        word.append(segments.pop(start))
                    segments.insert(start,"".join(word))
                    idx = start + 1 
            return
        if not flag:
            if len([ True for pattern in self.resources["patterns"] if pattern.match(segments[idx]) is not None ] ) > 0:
                start = idx
                flag = True
            self.adjust_seg(segments,idx=idx+1,start=start,flag=flag)
        else:
            if len([ True for pattern in self.resources["patterns"] if pattern.match("".join(segments[start:idx+1])) is not None ] ) > 0:
                self.adjust_seg(segments,idx=idx+1,start=start,flag=flag)
            else:
                
                loc = None
                # 避免 2007年金 --> 2007年/金
                if len(segments[idx]) > 1 and re.compile(r"^年金").match(segments[idx]) is  None:
                    for i in range(len(segments[idx])):
                        s = segments[idx][i]
                        if len([ True for pattern in self.resources["patterns"] if pattern.match("".join(segments[start:idx]) + s ) is not None ] ) == 0:
                            loc = i
                            break
                if loc is not None and loc != 0:
                    seg = segments[idx]
                    segments[idx] = seg[:loc]
                    idx += 1
                    segments.insert(idx,seg[loc:])
                
                if idx - start > 1:
                    word = []
                    for _ in range(start,idx):
                        word.append(segments.pop(start))
                    segments.insert(start,"".join(word))
                    idx = start + 1

                flag = False
                self.adjust_seg(segments,idx=idx,start=start+1,flag=flag)



    def getResources(self):
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"resources/kernel.txt"),"r",encoding="utf-8") as f:
            kernel = [w.strip() for w in f.readlines() if len(w.strip()) != 0 and not w.strip().startswith("#") ]

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"resources/limited-1.txt"),"r",encoding="utf-8") as f:
            limited = [w.strip() for w in f.readlines() if len(w.strip()) != 0 and not w.strip().startswith("#")  ]

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"resources/regex-limited-1.txt"),"r",encoding="utf-8") as f:
            regex_limited = [ re.compile(w.strip()) for w in f.readlines() if len(w.strip()) != 0 and not w.strip().startswith("#")  ]

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"resources/prefix.txt"),"r",encoding="utf-8") as f:
            prefix = [w.strip() for w in f.readlines() if len(w.strip()) != 0 and not w.strip().startswith("#")  ]

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"resources/suffix.txt"),"r",encoding="utf-8") as f:
            suffix = [w.strip() for w in f.readlines() if len(w.strip()) != 0 and not w.strip().startswith("#")  ]

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"resources/properties.txt"),"r",encoding="utf-8") as f:
            properties = [w.strip() for w in f.readlines() if len(w.strip()) != 0 and not w.strip().startswith("#")  ]

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"resources/standard-entity.txt"),"r",encoding="utf-8") as f:
            standard_entity = [w.strip() for w in f.readlines() if len(w.strip()) != 0 and not w.strip().startswith("#")  ]

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"resources/plans.txt"),"r",encoding="utf-8") as f:
            plans = [w.strip() for w in f.readlines() if len(w.strip()) != 0 and not w.strip().startswith("#")  ] 

        forward = suffix[:]
        forward += limited + plans
        backward = prefix[:]
        backward += limited + plans
        whole_without_kernel = list(set(prefix + limited + plans + suffix))
        invalid_single_word_as_entity = list(set(prefix + limited + suffix))
        skips = r".!/_,$%^*()?;；:-【】\"'\[\]——！，;:。？、~@#￥%……&*（）《》"
        patterns = [ 
             re.compile(r"^[A-Za-z]+\+{0,1}\s*款{0,1}$"),
             re.compile(r"^[一二三四五六七八九零〇]+\s*号{0,1}$"),
             re.compile(r"^[0-9]+\s*号{0,1}$"),
             re.compile(r"^[0-9]+\s*(年(?<!金))?(\d+月?)?(\d+(日(?<!额)|号)?)?((以|之)?(前|后))?$"),
             re.compile(r"^[0-9]+\s*(年(领)?)?$")
           ]
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"resources/detail/ambiguous_prefix.txt"),"r",encoding="utf-8") as f:
            ambiguous_prefix = [w.strip() for w in f.readlines() if len(w.strip()) != 0 and not w.strip().startswith("#") ]

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"resources/detail/ambiguous_suffix.txt"),"r",encoding="utf-8") as f:
            ambiguous_suffix = [w.strip() for w in f.readlines() if len(w.strip()) != 0 and not w.strip().startswith("#") ]

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"resources/detail/tail_plus_one_patterns.txt"),"r",encoding="utf-8") as f:
            tail_plus_one_patterns = [ re.compile(pattern) for pattern in f.readlines() if len(pattern.strip()) != 0 ]

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"resources/detail/tail_minus_one_and_tail_stop_patterns.txt"),"r",encoding="utf-8") as f:
            tail_minus_one_and_tail_stop_patterns = [ pattern for pattern in f.readlines() if len(pattern.strip()) != 0 ]

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"resources/detail/tail_minus_one_and_tail_cut_tail_patterns.txt"),"r",encoding="utf-8") as f:
            tail_minus_one_and_tail_cut_tail_patterns = [ pattern for pattern in f.readlines() if len(pattern.strip()) != 0 ]

        mytrie = marisa_trie.Trie([u'险'])

        result = {
                    "prefix":prefix,
                    "suffix":suffix,
                    "kernel":kernel,
                    "regex_limited":regex_limited,
                    "properties":properties,
                    "standard_entity":standard_entity,
                    "plans":plans,
                    "forward":forward,
                    "backward":backward,
                    "whole_without_kernel":whole_without_kernel,
                    "invalid_single_word_as_entity":invalid_single_word_as_entity,
                    "skips":skips,
                    "patterns":patterns,
                    "ambiguous_prefix":ambiguous_prefix,
                    "ambiguous_suffix":ambiguous_suffix,
                    "tail_plus_one_patterns":tail_plus_one_patterns,
                    "tail_minus_one_and_tail_stop_patterns":tail_minus_one_and_tail_stop_patterns,
                    "tail_minus_one_and_tail_cut_tail_patterns":tail_minus_one_and_tail_cut_tail_patterns,
                    "mytrie":mytrie,
                }

        return result
    