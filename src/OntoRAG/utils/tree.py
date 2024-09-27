 
def lemma(word):
    import spacy
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(word)
    lemmatized = tuple([token.lemma_ for token in doc])
    return lemmatized


class Tree:

    def __init__(self, value):
        self.value = lemma(value)
        self.synonyms = [value]
        self.count = 1
        self.children = []


    def __str_level__(self, level):
        res = ''
        for i in range(level):
            res += '  '
        res += f"{self.value} -> {self.count}\n"
        for c in self.children:
            res += c.__str_level__(level + 1)
        return res


    def __str__(self):
        res = ''
        for c in self.children:
            res += f"{c.synonyms[0]} isA {self.synonyms[0]}\n"
            res += c.__str__()
        return res


    def __repr__(self):
        return self.__str__()


    def add_child(self, node):
        if type(node) == str:
            node = Tree(node)
        for c in self.children:
            if c.value == node.value:
                return
        self.children.append(node)


    def get_terms(self, ctx=None):
        res = []
        for c in self.children:
            res += c.get_terms(ctx)
        if ctx is None:
            res += [self.synonyms[0]]
        else:
            found = []
            for syn in self.synonyms:
                if syn in ctx:
                    found.append(syn)
                    break
            if len(found) == 0:
                res += [self.synonyms[0]]
            else:
                res += found
        return res
    

    def get_lemmatized_wordmap(self):
        res = {}
        for c in self.children:
            res = {**res, **c.get_lemmatized_wordmap()}
        res[self.value] = self
        return res


    def clone(self):
        res = Tree(self.synonyms[0])
        res.value = self.value
        res.synonyms = self.synonyms
        res.children = [c.clone() for c in self.children]
        return res
    

    def get_nodes_list(self):
        res = []
        for c in self.children:
            res += c.get_nodes_list()
        res.append(self)
        return res
    

    def get_level_terms(self, level, ctx=None):
        if level == 0:
            if ctx is None:
                return [self.synonyms[0]]
            found = []
            for syn in self.synonyms:
                if syn in ctx:
                    found.append(syn)
                    break
            if len(found) == 0:
                return [self.synonyms[0]]
            return found
                        
        res = []
        for c in self.children:
            res += c.get_level_terms(level - 1, ctx)
        return res
    

    def prune_to_level(self, level):
        if level == 0:
            self.children = []
            return
        for c in self.children:
            c.prune_to_level(level - 1)


    def exists(self, term):
        if term in self.synonyms:
            return True
        for c in self.children:
            if c.exists(term):
                return True
        return False
    

    def get_node(self, term):
        if term in self.synonyms:
            return self
        for c in self.children:
            res = c.get_node(term)
            if res:
                return res
        return None
    
    def remove_node(self, term):

        if type(term) == str:
            for c in self.children:
                if term in c.synonyms:
                    self.children.remove(c)
                c.remove_node(term) 
        elif type(term) == Tree:
            for c in self.children:
                if term in c.children:
                    c.children.remove(term)
                c.remove_node(term)
        else:
            raise ValueError("term should be a string or a Tree object")


    def _get_ctx_term(self, synonyms, ctx=None):
        if ctx is None:
            return synonyms[0]
        for syn in synonyms:
            if syn in ctx:
                return syn
        return synonyms[0]


    def _get_level_terms_and_path(self, level, visited, ctx=None):
        if self in visited:
            return [], []
        visited.append(self)
        # returns two lists: (1) the terms at the level and (2) the path to the terms as a list
        if level == 0:
            term = self._get_ctx_term(self.synonyms, ctx)
            return [term], [[term]]
        else:
            res = []
            path = []
            for c in self.children:
                terms, p = c._get_level_terms_and_path(level - 1, visited, ctx)
                res += terms
                for p_ in p:
                    path.append([self._get_ctx_term(self.synonyms, ctx)] + p_)
            return res, path


    def get_level_terms_and_path(self, level, ctx=None):
        return self._get_level_terms_and_path(level, [], ctx)


    def _get_terms_and_paths(self, visited, ctx=None):

        if self in visited:
            return [], []
        visited.append(self)

        # returns two lists: (1) the terms and (2) the path to the terms
        res = []
        path = []

        for c in self.children:
            terms, p = c._get_terms_and_paths(visited, ctx)
            res += terms
            for p_ in p:
                path.append([self._get_ctx_term(self.synonyms, ctx)] + p_)
        
        term = self._get_ctx_term(self.synonyms, ctx)
        res.append(term)
        path.append([term])
        return res, path


    def get_terms_and_paths(self, ctx=None):
        return self._get_terms_and_paths([], ctx)
        

    def _get_all_childs_and_self(self):
        res = [self]
        for c in self.children:
            res += c._get_all_childs_and_self()
        return res


    def list_all_childs(self):
        # list only the children BELOW the current node, not including the current node
        res = []
        for c in self.children:
            res += c._get_all_childs_and_self()
        return res

    
    def prune_hierarchy_repeated_nodes(self):
        # remove a node if the same node is present somewhere below in the hierarchy
        for c in self.children:
            c.prune_hierarchy_repeated_nodes()
        to_remove = []
        all_childs = []
        for c in self.children:
            all_childs += c.list_all_childs()
        for c in self.children:
            if c in all_childs: # child c is also present somewhere below (i.e. it is a child of a child)
                to_remove.append(c)
        for c in to_remove:
            self.children.remove(c)


    def _get_all_ancestors(self, current_tree, current_ancestors, target_tree):
        if current_tree == target_tree:
            return current_ancestors
        res = []
        for c in current_tree.children:
            res += c._get_all_ancestors(c, current_ancestors + [current_tree], target_tree)
        return res


    def get_all_ancestors(self, tree):
        return self._get_all_ancestors(self, [], tree)


    def get_leaf_nodes(self):
        res = []
        for c in self.children:
            res += c.get_leaf_nodes()
        if len(self.children) == 0:
            res.append(self)
        return res


    def get_depth(self):
        if len(self.children) == 0:
            return 1
        return 1 + max([c.get_depth() for c in self.children])

                


    