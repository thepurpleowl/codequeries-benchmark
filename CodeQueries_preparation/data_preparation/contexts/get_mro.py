from collections import deque
from itertools import islice
from typing import List, Dict, Tuple, Optional


class LinearizationDeque(deque):
    """
    A deque to represent linearization of a class
    """
    @property
    def head(self) -> Optional[type]:
        """
        Return head of deque
        """
        try:
            return self[0]
        except IndexError:
            return None

    @property
    def tail(self) -> islice:  # type: ignore
        """
        Return islice object, which is suffice for iteration or calling `in`
        """
        try:
            return islice(self, 1, self.__len__())
        except (ValueError, IndexError):
            return islice([], 0, 0)


class LinearizationDequeList:
    """
    A class represents list of linearizations (dependencies)
    The last element of LinearizationDequeList is a list of parents.
    It's needed for the merge process preserves the local
    precedence order of direct parent classes.
    """
    def __init__(self, *lists: Tuple[List[type]]) -> None:
        self._lists = [LinearizationDeque(i) for i in lists]

    def __contains__(self, item: type) -> bool:
        """
        Return True if any linearization's tail contains an item
        """
        return any([item in dep_list.tail for dep_list in self._lists])

    def __len__(self):
        size = len(self._lists)
        return (size - 1) if size else 0

    def __repr__(self):
        return self._lists.__repr__()

    @property
    def heads(self) -> List[Optional[type]]:
        return [h.head for h in self._lists]

    @property
    def tails(self) -> 'LinearizationDequeList':
        """
        Return self so that __contains__ could be called
        Used for readability reasons only
        """
        return self

    @property
    def exhausted(self) -> bool:
        """
        Return True if all elements of the lists are exhausted
        """
        return all(map(lambda x: len(x) == 0, self._lists))

    def remove(self, item: Optional[type]) -> None:
        """
        Remove head from all LinearizationDeque
        """
        for i in self._lists:
            if i and i.head == item:
                i.popleft()


def _merge(*lists) -> list:
    """
    Return self so that __contains__ could be called
    Used for readability reasons only
    Args:
        A list of LinearizationDeque
    Returns:
        A list of classes in order corresponding to Python's MRO
    """
    result: List[Optional[type]] = []
    linearizations = LinearizationDequeList(*lists)

    while True:
        if linearizations.exhausted:
            return result

        for head in linearizations.heads:
            if head and (head not in linearizations.tails):
                result.append(head)
                linearizations.remove(head)

                # Once candidate added to result, next
                # candidate selection iteration starts
                break
        else:
            # Loop never broke, no linearization could possibly be found
            raise ValueError('Cannot compute linearization, a cycle found')


def mro(bases_dict: Dict, cls: str):
    """
    Return mro as per c3 linearization
    Args:
        bases_dict: dict with classes as key and class.__bases__ as value
        cls: target class
    Returns:
        A list of classes in order corresponding to Python's MRO
    """
    all_class_names = bases_dict.keys()
    result = [cls]

    if not bases_dict[cls]:
        return result
    else:
        return (result
                + _merge(*[mro(bases_dict, kls) for kls in bases_dict[cls] if kls in all_class_names],
                         bases_dict[cls]))

# #example
# cc = {'A': [], 'B': ['A'],'L': [], 'C': ['A', 'L'],
#       'D': ['C'],' E': ['B', 'C'], 'F': ['D', 'B']}

# print(mro(cc, 'F'))
