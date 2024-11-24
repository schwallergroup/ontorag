import spacy

nlp = spacy.load("en_core_web_sm")


def lemma(word):
    doc = nlp(word)
    lemmatized = tuple([token.lemma_ for token in doc])
    return lemmatized


class Tree:

    def __init__(self, value):
        """
        Initialize a Tree object with a value.

        Parameters:
        value (str): The value of the node.
        """
        self.value = lemma(value)
        self.synonyms = [value]
        self.count = 1
        self.children = []

    def __str_level__(self, level):
        """
        String representation of the tree with indentation according to the level.
        """
        res = ""
        for i in range(level):
            res += "  "
        res += f"{self.value} -> {self.count}\n"
        for c in self.children:
            res += c.__str_level__(level + 1)
        return res

    def __str__(self):
        """
        String representation of the tree formatted with 'isA' relations.
        """
        res = ""
        for c in self.children:
            res += f"{c.synonyms[0]} isA {self.synonyms[0]}\n"
            res += c.__str__()
        return res

    def __repr__(self):
        return self.__str__()

    def add_child(self, node):
        """
        Add a child to the current node.

        Parameters:
        node (str or Tree): The child to add to the current node.
        """
        if type(node) == str:
            node = Tree(node)
        for c in self.children:
            if c.value == node.value:
                return
        self.children.append(node)

    def get_terms(self, ctx=None):
        """
        Get all the terms in the tree. If context is provided, return the synonym that appears in the context, otherwise return the first synonym.


        Parameters:
        ctx (list, optional): A list of terms to search for in the synonyms.

        Returns:
        list: List of terms in the tree.
        """
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
        """
        Get a dictionary with the lemmatized words as keys and the corresponding Tree object as values.
        """
        res = {}
        for c in self.children:
            res = {**res, **c.get_lemmatized_wordmap()}
        res[self.value] = self
        return res

    def clone(self):
        """
        Clone the current tree.
        """
        res = Tree(self.synonyms[0])
        res.value = self.value
        res.synonyms = self.synonyms
        res.children = [c.clone() for c in self.children]
        return res

    def get_nodes_list(self):
        """
        Get a list of all the nodes in the tree.
        """
        res = []
        for c in self.children:
            res += c.get_nodes_list()
        res.append(self)
        return res

    def get_level_terms(self, level, ctx=None):
        """
        Get the terms at a specific level in the tree. If context is provided, return the synonym that appears in the context, otherwise return the first synonym.

        Parameters:
        level (int): The level to search for.
        ctx (list, optional): A list of terms to search for in the synonyms.

        Returns:
        list: List of terms at the specified level.
        """
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
        """
        Prune the tree to a specific level.

        Parameters:
        level (int): The level to prune the tree to.
        """
        if level == 0:
            self.children = []
            return
        for c in self.children:
            c.prune_to_level(level - 1)

    def exists(self, term):
        """
        Return True if the term exists in the tree.

        Parameters:
        term (str): The term to search for.
        """
        if term in self.synonyms:
            return True
        for c in self.children:
            if c.exists(term):
                return True
        return False

    def get_node(self, term):
        """
        Get the node with the term. If the term is not found, return None.

        Parameters:
        term (str): The term to search for.
        """
        if term in self.synonyms:
            return self
        for c in self.children:
            res = c.get_node(term)
            if res:
                return res
        return None

    def remove_node(self, term):
        """
        Remove a node from the tree.

        Parameters:
        term (str or Tree): The term to remove.
        """
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
        """
        Get the term that appears in the context. If no context is provided, return the first synonym.

        Parameters:
        synonyms (list): List of synonyms.
        ctx (list, optional): A list of terms to search for in the synonyms.
        """
        if ctx is None:
            return synonyms[0]
        for syn in synonyms:
            if syn in ctx:
                return syn
        return synonyms[0]

    def _get_level_terms_and_path(self, level, visited, ctx=None):
        """
        Auxiliary function to get the terms at a specific level in the tree and the path from root to the terms.
        If context is provided, return the synonym that appears in the context, otherwise return the first synonym.

        Parameters:
        level (int): The level to search for.
        visited (list): List of visited nodes.
        ctx (list, optional): A list of terms to search for in the synonyms.

        Returns:
        list: List of terms at the specified level.
        list: List of paths to the terms.
        """
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
        """
        Get the terms at a specific level in the tree and the path from root to the terms.
        If context is provided, return the synonym that appears in the context, otherwise return the first synonym.

        Parameters:
        level (int): The level to search for.
        ctx (list, optional): A list of terms to search for in the synonyms.

        Returns:
        list: List of terms at the specified level.
        list: List of paths to the terms.
        """
        return self._get_level_terms_and_path(level, [], ctx)

    def _get_terms_and_paths(self, visited, ctx=None):
        """
        Auxiliary function to get all the terms in the tree and the path from root to the terms.
        If context is provided, return the synonym that appears in the context, otherwise return the first synonym.

        Parameters:
        visited (list): List of visited nodes.
        ctx (list, optional): A list of terms to search for in the synonyms.

        Returns:
        list: List of terms in the tree.
        list: List of paths to the terms.
        """
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
        """
        Get all the terms in the tree and the path from root to the terms.
        If context is provided, return the synonym that appears in the context, otherwise return the first synonym.

        Parameters:
        ctx (list, optional): A list of terms to search for in the synonyms.

        Returns:
        list: List of terms in the tree.
        list: List of paths to the terms.
        """
        return self._get_terms_and_paths([], ctx)

    def _get_all_childs_and_self(self):
        """
        Auxiliary function to get all the children below the current node, including the current node.
        """
        res = [self]
        for c in self.children:
            res += c._get_all_childs_and_self()
        return res

    def list_all_childs(self):
        """
        Get all the children below the current node, NOT including the current node.
        """
        # list only the children BELOW the current node, not including the current node
        res = []
        for c in self.children:
            res += c._get_all_childs_and_self()
        return res

    def prune_hierarchy_repeated_nodes(self):
        """
        Remove a node if the same node is present somewhere below in the hierarchy
        """
        for c in self.children:
            c.prune_hierarchy_repeated_nodes()
        to_remove = []
        all_childs = []
        for c in self.children:
            all_childs += c.list_all_childs()
        for c in self.children:
            if (
                c in all_childs
            ):  # child c is also present somewhere below (i.e. it is a child of a child)
                to_remove.append(c)
        for c in to_remove:
            self.children.remove(c)

    def _get_all_ancestors(self, current_tree, current_ancestors, target_tree):
        """
        Auxiliary function to get all the ancestors of a target tree.

        Parameters:
        current_tree (Tree): The current tree.
        current_ancestors (list): List of current ancestors.
        target_tree (Tree): The target tree.
        """
        if current_tree == target_tree:
            return current_ancestors
        res = []
        for c in current_tree.children:
            res += c._get_all_ancestors(
                c, current_ancestors + [current_tree], target_tree
            )
        return res

    def get_all_ancestors(self, tree):
        """
        Get all the ancestors of a target tree.

        Parameters:
        tree (Tree): The target tree.
        """
        return self._get_all_ancestors(self, [], tree)

    def get_leaf_nodes(self):
        """
        Get all the leaf nodes in the tree.
        """
        res = []
        for c in self.children:
            res += c.get_leaf_nodes()
        if len(self.children) == 0:
            res.append(self)
        return res

    def get_depth(self):
        """
        Get the depth of the tree.
        """
        if len(self.children) == 0:
            return 1
        return 1 + max([c.get_depth() for c in self.children])
