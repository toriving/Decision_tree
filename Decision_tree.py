import math
from collections import Counter, defaultdict
from functools import partial

def calc_entropy(class_prob):
    # 각 클래스에 속할 확률을 인수로 받아 entropy 계산, 0 은 제외
    return sum(-p * math.log(p, 2) for p in class_prob if p is not 0)

def class_prob(labels):
    # 클래스에 속할 확률 계산
    total_num = len(labels)
    # counter(labels) = {class1 : 2, class2 : 3} -> values() -> 2,3 -> prob
    return [float(num) / float(total_num) for num in Counter(labels).values()]

def data_entropy(data):
    # data들의 entropy 계산
    labels = [label for _, label in data] # True, False / label 만 저장
    prob = class_prob(labels)
    return calc_entropy(prob)

def subset_entropy(subsets):
    # subset은 레이블이 있는 데이터의 list의 list (Tree 구조)
    total_num = sum(len(subset) for subset in subsets)
    # 그에 대한 엔트로피를 계산한 뒤 모든 subset의 엔트로피 합친 값 반환
    # subset A의 엔트로피는 A 요소별 엔트로피의 합 * A의 영역 비율
    return sum(data_entropy(subset) * len(subset) / total_num for subset in subsets)

def make_subset(inputs, feature):
    # feature 기준으로 group을 만듬.
    # feature 가 3개면 그룹수 = 3 ex) level : senior, mid, junior -> 3
    # senior : ~~~~, ~~~, ~~~ , mid : ~~~,~~~,~~~ ...
    groups = defaultdict(list)
    for input in inputs:
        key = input[0][feature]
        groups[key].append(input)
    return groups

def total_entropy(inputs, feature):
    # subset entropy 계산
    groups = make_subset(inputs, feature)
    return subset_entropy(groups.values())

def build(inputs, split_candidates=None):
    # 첫 분기면 모든 변수가 후보
    if split_candidates is None:
        split_candidates = inputs[0][0].keys()

    # 입력데이터에서 범주별 개수를 센다.
    num_inputs = len(inputs)
    num_class_T = len([label for _, label in inputs if label])
    num_class_F = num_inputs - num_class_T

    if num_class_T == 0: return False  # true 없으면 false leaf 반환
    if num_class_F == 0: return True  # ''

    # subset 기준으로 사용할 변수가 없으면 (맨위에서 한번해준후.)
    if not split_candidates:
        # 다수결
        return num_class_T >= num_class_F

    # 아니면 가장 적합한 변수로 분기
    best_attribute = min(split_candidates, key=partial(total_entropy, inputs))
    subsets = make_subset(inputs, best_attribute)
    new_candidates = [c for c in split_candidates if c != best_attribute]

    # 재귀적으로 서브트리를 구축
    subtrees = {attribute_value : build(subset, new_candidates) for attribute_value, subset in subsets.items()}

    # default
    subtrees[None] = num_class_T > num_class_F

    return (best_attribute, subtrees)

def classify(tree, input):
    # 주어진 tree를 기준으로 input을 분류
    # leaf 노드이면 값 반환
    if tree in [True, False]:
        return tree

    # 그게 아니면 데이터의 변수로 분기
    # 키로 변수값, 값으로 서브트리를 나타내는 dict 사용
    attribute, subtree_dict = tree

    # 만약 입력된 데이터 변수 가운데 하나가
    # 기존에 관찰되지 않았다면 None
    subtree_key = input.get(attribute)

    # 키에 해당하는 서브트리가 존재하지 않을 때
    if subtree_key not in subtree_dict:
        # None 서브트리를 사용
        subtree_key = None

    # 적절한 서브트리를 선택
    subtree = subtree_dict[subtree_key]
    # 그리고 입력된 데이터를 분류
    return classify(subtree, input)

def main():
    # training data
    inputs = [
        ({'level': 'Senior', 'lang': 'Java', 'tweets': 'no', 'phd': 'no'}, False),
        ({'level': 'Senior', 'lang': 'Java', 'tweets': 'no', 'phd': 'yes'}, False),
        ({'level': 'Mid', 'lang': 'Python', 'tweets': 'no', 'phd': 'no'}, True),
        ({'level': 'Junior', 'lang': 'Python', 'tweets': 'no', 'phd': 'no'}, True),
        ({'level': 'Junior', 'lang': 'R', 'tweets': 'yes', 'phd': 'no'}, True),
        ({'level': 'Junior', 'lang': 'R', 'tweets': 'yes', 'phd': 'yes'}, False),
        ({'level': 'Mid', 'lang': 'R', 'tweets': 'yes', 'phd': 'yes'}, True),
        ({'level': 'Senior', 'lang': 'Python', 'tweets': 'no', 'phd': 'no'}, False),
        ({'level': 'Senior', 'lang': 'R', 'tweets': 'yes', 'phd': 'no'}, True),
        ({'level': 'Junior', 'lang': 'Python', 'tweets': 'yes', 'phd': 'no'}, True),
        ({'level': 'Senior', 'lang': 'Python', 'tweets': 'yes', 'phd': 'yes'}, True),
        ({'level': 'Mid', 'lang': 'Python', 'tweets': 'no', 'phd': 'yes'}, True),
        ({'level': 'Mid', 'lang': 'Java', 'tweets': 'yes', 'phd': 'no'}, True),
        ({'level': 'Junior', 'lang': 'Python', 'tweets': 'no', 'phd': 'yes'}, False)
    ]

    tree = build(inputs)
    print(classify(tree,
                   {"level": "Junior",
                    "lang": "Java",
                    "tweets": "yes",
                    "phd": "no"}))  # -> True
    print(classify(tree,
                   {"level": "Junior",
                    "lang": "Java",
                    "tweets": "yes",
                    "phd": "yes"}))  # -> False
    print(classify(tree,
                   {"level": "Mid",
                    "lang": "Java",
                    "tweets": "yes",
                    "phd": "yes"}))  # -> True
    print(classify(tree,
                   {"level": "None",
                    "lang": "Java",
                    "tweets": "yes",
                    "phd": "yes"}))  # -> True
    print(classify(tree,
                   {"level": "Senior",
                    "lang": "Java",
                    "tweets": "No",
                    "phd": "yes"}))  # -> False
    print(classify(tree,
                   {"level": "Senior",
                    "lang": "Java",
                    "tweets": "yes",
                    "phd": "yes"}))  # -> True


if __name__ == "__main__":
    main()