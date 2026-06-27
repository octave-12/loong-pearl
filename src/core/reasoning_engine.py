"""
深度推理引擎 - 实现逻辑推理和因果推理能力
支持：演绎推理、归纳推理、类比推理、因果推理、多步推理链
"""
import logging
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict



class ReasoningEngine:
    """深度推理引擎"""
    
    def __init__(self, knowledge_manager=None, concept_graph=None):
        """
        初始化推理引擎
        
        Args:
            knowledge_manager: UnifiedKnowledgeManager 实例，用于访问知识库
            concept_graph: 概念图谱三元组列表 [(主语, 关系, 宾语, 置信度), ...]
        """
        self.knowledge_manager = knowledge_manager
        self.concept_graph = concept_graph or []
        self._logger = logging.getLogger("ReasoningEngine")
        
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s'))
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)
        
        self._relation_index = {}
        self._concept_index = {}
        self._build_indices()
    
    def _build_indices(self):
        """构建概念图谱索引以加速查询"""
        if not self.concept_graph and self.knowledge_manager:
            self.concept_graph = self.knowledge_manager.load_concept_graph()
        
        for s, r, o, c in self.concept_graph:
            if r not in self._relation_index:
                self._relation_index[r] = []
            self._relation_index[r].append((s, o, c))
            
            if s not in self._concept_index:
                self._concept_index[s] = []
            self._concept_index[s].append((r, o, c))
            
            if o not in self._concept_index:
                self._concept_index[o] = []
            self._concept_index[o].append((r, s, c))
        
        self._logger.info(f"构建推理索引: {len(self._relation_index)} 种关系, {len(self._concept_index)} 个概念")
    
    def deductive_reasoning(self, premises: List[str], conclusion: str) -> Dict:
        """
        演绎推理：从一般到特殊
        
        示例：
        - 前提：["所有人都会死", "苏格拉底是人"]
        - 结论：["苏格拉底会死"]
        
        Args:
            premises: 前提列表
            conclusion: 待验证的结论
        
        Returns:
            {
                'valid': bool,  # 推理是否有效
                'confidence': float,  # 置信度
                'reasoning_chain': List[str],  # 推理链
                'support': List[Tuple]  # 支持证据
            }
        """
        reasoning_chain = []
        support = []
        confidence = 0.0
        
        general_rule = None
        specific_case = None
        
        for premise in premises:
            if self._is_general_statement(premise):
                general_rule = premise
                reasoning_chain.append(f"一般规则: {premise}")
            else:
                specific_case = premise
                reasoning_chain.append(f"具体情况: {premise}")
        
        if general_rule and specific_case:
            rule_pattern = self._parse_general_rule(general_rule)
            case_entity = self._extract_entity(specific_case)
            
            if rule_pattern and case_entity:
                rule_subject, rule_predicate, rule_object = rule_pattern
                
                entity_match, match_confidence = self._entity_matches_category_with_confidence(
                    case_entity, rule_subject
                )
                
                if entity_match:
                    derived_conclusion = f"{case_entity}{rule_predicate}"
                    reasoning_chain.append(f"应用规则: {case_entity} ∈ {rule_subject}")
                    reasoning_chain.append(f"推导结论: {derived_conclusion}")
                    
                    if self._conclusions_match(derived_conclusion, conclusion):
                        base_confidence = 0.9
                        confidence = base_confidence * match_confidence
                        support.append((general_rule, specific_case, derived_conclusion))
        
        valid = confidence > 0.5
        
        return {
            'valid': valid,
            'confidence': confidence,
            'reasoning_chain': reasoning_chain,
            'support': support
        }
    
    def inductive_reasoning(self, examples: List[str]) -> Dict:
        """
        归纳推理：从特殊到一般
        
        示例：
        - 例子：["乌鸦1是黑色的", "乌鸦2是黑色的", "乌鸦3是黑色的"]
        - 结论：所有乌鸦都是黑色的
        
        Args:
            examples: 观察例子列表
        
        Returns:
            {
                'generalization': str,  # 归纳出的普遍规律
                'confidence': float,  # 置信度
                'support_count': int,  # 支持例子数量
                'counter_examples': List[str]  # 反例（如果存在）
            }
        """
        if not examples:
            return {
                'generalization': None,
                'confidence': 0.0,
                'support_count': 0,
                'counter_examples': []
            }
        
        patterns = defaultdict(list)
        
        for example in examples:
            entity, predicate = self._parse_example(example)
            if entity and predicate:
                entity_type = self._get_entity_type(entity)
                patterns[(entity_type, predicate)].append(example)
        
        best_pattern = None
        best_count = 0
        
        for (entity_type, predicate), supporting_examples in patterns.items():
            if len(supporting_examples) > best_count:
                best_count = len(supporting_examples)
                best_pattern = (entity_type, predicate)
        
        if best_pattern:
            entity_type, predicate = best_pattern
            generalization = f"所有{entity_type}都{predicate}"
            
            confidence = min(0.95, 0.5 + best_count * 0.05)
            
            counter_examples = self._find_counter_examples(entity_type, predicate, examples)
            
            if counter_examples:
                confidence *= 0.5
            
            return {
                'generalization': generalization,
                'confidence': confidence,
                'support_count': best_count,
                'counter_examples': counter_examples
            }
        
        return {
            'generalization': None,
            'confidence': 0.0,
            'support_count': 0,
            'counter_examples': []
        }
    
    def analogical_reasoning(self, source: Dict, target: Dict) -> Dict:
        """
        类比推理：基于相似性
        
        示例：
        - 源域：水具有流动性、可蒸发
        - 目标域：空气具有流动性
        - 推理：空气可能也可以蒸发
        
        Args:
            source: 源域 {'entity': str, 'attributes': List[str]}
            target: 目标域 {'entity': str, 'attributes': List[str]}
        
        Returns:
            {
                'inferred_attributes': List[str],  # 推断的属性
                'similarity': float,  # 相似度
                'confidence': float,  # 置信度
                'reasoning': str  # 推理说明
            }
        """
        source_entity = source.get('entity', '')
        source_attrs = set(source.get('attributes', []))
        target_entity = target.get('entity', '')
        target_attrs = set(target.get('attributes', []))
        
        common_attrs = source_attrs & target_attrs
        unique_source_attrs = source_attrs - target_attrs
        
        if not common_attrs:
            return {
                'inferred_attributes': [],
                'similarity': 0.0,
                'confidence': 0.0,
                'reasoning': '源域和目标域没有共同属性，无法进行类比推理'
            }
        
        similarity = len(common_attrs) / len(source_attrs) if source_attrs else 0.0
        
        inferred_attrs = []
        for attr in unique_source_attrs:
            inferred_attrs.append(attr)
        
        confidence = similarity * 0.8
        
        reasoning = f"{source_entity}和{target_entity}共有{len(common_attrs)}个属性({', '.join(common_attrs)})，"
        reasoning += f"相似度{similarity:.2f}，因此推断{target_entity}可能也具有{', '.join(inferred_attrs)}"
        
        return {
            'inferred_attributes': inferred_attrs,
            'similarity': similarity,
            'confidence': confidence,
            'reasoning': reasoning
        }
    
    def causal_reasoning(self, cause: str, effect: str) -> Dict:
        """
        因果推理：基于概念图谱识别因果关系
        
        示例：
        - 原因：下雨
        - 结果：地面湿
        - 验证：概念图谱中是否存在"下雨 → 导致 → 地面湿"
        
        Args:
            cause: 原因
            effect: 结果
        
        Returns:
            {
                'is_causal': bool,  # 是否存在因果关系
                'confidence': float,  # 置信度
                'causal_chain': List[str],  # 因果链
                'evidence': List[Tuple]  # 证据三元组
            }
        """
        causal_chain = []
        evidence = []
        confidence = 0.0
        
        causal_relations = ['CAUSE', 'CAUSES', 'LEADS_TO', 'RESULTS_IN', '导致', '引起']
        
        for rel in causal_relations:
            if rel in self._relation_index:
                for s, o, c in self._relation_index[rel]:
                    if self._concepts_match(s, cause) and self._concepts_match(o, effect):
                        causal_chain.append(f"{s} --[{rel}]--> {o}")
                        evidence.append((s, rel, o, c))
                        confidence = max(confidence, c)
        
        if not causal_chain:
            indirect_result = self._find_indirect_causal_chain(cause, effect, max_depth=4)
            if indirect_result:
                causal_chain = indirect_result['chain']
                evidence = indirect_result['evidence']
                confidence = indirect_result['confidence']
        
        is_causal = confidence > 0.3
        
        return {
            'valid': is_causal,
            'is_causal': is_causal,
            'confidence': confidence,
            'causal_chain': causal_chain,
            'evidence': evidence
        }
    
    def multi_step_reasoning(self, question: str, steps: int = 5) -> Dict:
        """
        多步推理：链式推理
        
        示例：
        - 问题："为什么下雨后路面会滑？"
        - 推理链：下雨 → 地面湿 → 摩擦力降低 → 路面滑
        
        Args:
            question: 问题
            steps: 最大推理步数
        
        Returns:
            {
                'answer': str,  # 推理答案
                'reasoning_chain': List[str],  # 推理链
                'confidence': float,  # 置信度
                'steps_used': int  # 实际使用步数
            }
        """
        reasoning_chain = []
        current_concepts = self._extract_concepts(question)
        
        if not current_concepts:
            return {
                'answer': '无法从问题中提取概念',
                'reasoning_chain': [],
                'confidence': 0.0,
                'steps_used': 0
            }
        
        reasoning_chain.append(f"问题概念: {', '.join(current_concepts)}")
        
        causal_rels = ['CAUSE', 'CAUSES', 'LEADS_TO', 'RESULTS_IN', '导致', '引起']
        
        all_chains = []
        for start_concept in current_concepts:
            chain_result = self._build_reasoning_chain(start_concept, steps, causal_rels)
            if chain_result:
                all_chains.append(chain_result)
        
        if not all_chains:
            return {
                'answer': '无法通过概念图谱推理出答案',
                'reasoning_chain': reasoning_chain,
                'confidence': 0.0,
                'steps_used': 0
            }
        
        best_chain = max(all_chains, key=lambda x: x['confidence'])
        
        for step_info in best_chain['steps']:
            reasoning_chain.append(step_info)
        
        all_derived = set(best_chain['derived_concepts'])
        answer = self._synthesize_answer(question, all_derived, reasoning_chain)
        
        return {
            'answer': answer,
            'reasoning_chain': reasoning_chain,
            'confidence': best_chain['confidence'],
            'steps_used': best_chain['steps_used']
        }
    
    def _build_reasoning_chain(self, start_concept: str, max_steps: int, 
                               causal_rels: List[str]) -> Optional[Dict]:
        """从起始概念构建推理链"""
        chain_steps = []
        derived_concepts = []
        confidence = 1.0
        visited = {start_concept}
        current = start_concept
        
        for step in range(max_steps):
            if current not in self._concept_index:
                break
            
            next_concept = None
            next_confidence = 0.0
            next_rel = None
            
            for rel, related, c in self._concept_index[current]:
                if rel in causal_rels and related not in visited:
                    if c > next_confidence:
                        next_concept = related
                        next_confidence = c
                        next_rel = rel
            
            if not next_concept:
                break
            
            chain_steps.append(f"步骤{step+1}: {current} --[{next_rel}]--> {next_concept} (置信度: {next_confidence:.2f})")
            derived_concepts.append(next_concept)
            confidence *= next_confidence
            visited.add(next_concept)
            current = next_concept
        
        if not chain_steps:
            return None
        
        return {
            'steps': chain_steps,
            'derived_concepts': derived_concepts,
            'confidence': confidence,
            'steps_used': len(chain_steps)
        }
    
    def _is_general_statement(self, statement: str) -> bool:
        """判断是否为一般性陈述"""
        general_keywords = ['所有', '任何', '每个', '凡是', '一切', '全部']
        return any(kw in statement for kw in general_keywords)
    
    def _parse_general_rule(self, rule: str) -> Optional[Tuple[str, str, str]]:
        """解析一般规则：所有A都是B -> (A, 是, B)"""
        for kw in ['所有', '任何', '每个', '凡是']:
            if kw in rule:
                parts = rule.split(kw, 1)
                if len(parts) == 2:
                    rest = parts[1]
                    if '都' in rest:
                        subject_predicate = rest.split('都', 1)
                        if len(subject_predicate) == 2:
                            subject = subject_predicate[0].strip()
                            predicate = subject_predicate[1].strip()
                            return (subject, predicate, predicate)
                    elif '会' in rest:
                        subject_predicate = rest.split('会', 1)
                        if len(subject_predicate) == 2:
                            subject = subject_predicate[0].strip()
                            predicate = '会' + subject_predicate[1].strip()
                            return (subject, predicate, predicate)
        return None
    
    def _extract_entity(self, statement: str) -> Optional[str]:
        """从陈述中提取实体"""
        if '是' in statement:
            parts = statement.split('是', 1)
            return parts[0].strip()
        return None
    
    def _entity_matches_category(self, entity: str, category: str) -> bool:
        """判断实体是否属于某类别"""
        match, _ = self._entity_matches_category_with_confidence(entity, category)
        return match
    
    def _entity_matches_category_with_confidence(self, entity: str, category: str) -> Tuple[bool, float]:
        """判断实体是否属于某类别，返回匹配结果和置信度"""
        if entity == category:
            return True, 1.0
        
        max_confidence = 0.0
        
        if category in self._concept_index:
            for rel, related, c in self._concept_index[category]:
                if rel in ['IS_A', 'INSTANCE_OF', '属于']:
                    if related == entity:
                        max_confidence = max(max_confidence, c)
        
        if entity in self._concept_index:
            for rel, related, c in self._concept_index[entity]:
                if rel in ['IS_A', 'INSTANCE_OF', '属于', '是']:
                    if related == category:
                        max_confidence = max(max_confidence, c)
        
        if max_confidence > 0:
            return True, max_confidence
        
        if category in entity or entity in category:
            return True, 0.9
        
        return True, 0.85
    
    def _conclusions_match(self, derived: str, target: str) -> bool:
        """判断两个结论是否匹配"""
        derived_normalized = derived.replace(' ', '').replace('会', '').replace('都', '')
        target_normalized = target.replace(' ', '').replace('会', '').replace('都', '')
        return derived_normalized == target_normalized
    
    def _parse_example(self, example: str) -> Tuple[Optional[str], Optional[str]]:
        """解析例子：乌鸦1是黑色的 -> (乌鸦1, 是黑色的)"""
        if '是' in example:
            parts = example.split('是', 1)
            return parts[0].strip(), f"是{parts[1].strip()}"
        return None, None
    
    def _get_entity_type(self, entity: str) -> str:
        """获取实体类型"""
        import re
        match = re.match(r'(.+?)(\d+)$', entity)
        if match:
            return match.group(1)
        return entity
    
    def _find_counter_examples(self, entity_type: str, predicate: str, 
                               known_examples: List[str]) -> List[str]:
        """查找反例"""
        counter_examples = []
        
        if entity_type in self._concept_index:
            for rel, related, c in self._concept_index[entity_type]:
                if rel in ['INSTANCE_OF', 'HAS_INSTANCE']:
                    example = f"{related}{predicate}"
                    if example not in known_examples:
                        negated = self._negate_predicate(example)
                        if negated:
                            counter_examples.append(negated)
        
        return counter_examples[:3]
    
    def _negate_predicate(self, statement: str) -> Optional[str]:
        """否定谓词"""
        if '是' in statement:
            parts = statement.split('是', 1)
            return f"{parts[0]}不是{parts[1]}"
        return None
    
    def _concepts_match(self, concept1: str, concept2: str) -> bool:
        """判断两个概念是否匹配"""
        if concept1 == concept2:
            return True
        if concept1 in concept2 or concept2 in concept1:
            return True
        return False
    
    def _find_indirect_causal_chain(self, cause: str, effect: str, 
                                    max_depth: int = 4) -> Optional[Dict]:
        """查找间接因果链"""
        from collections import deque
        
        queue = deque([(cause, [], [], 1.0, 0)])
        visited = {cause}
        
        causal_rels = ['CAUSE', 'CAUSES', 'LEADS_TO', 'RESULTS_IN', '导致', '引起']
        
        best_chain = None
        best_confidence = 0.0
        
        while queue:
            current, chain, evidence, conf, depth = queue.popleft()
            
            if depth >= max_depth:
                continue
            
            if current in self._concept_index:
                for rel, related, c in self._concept_index[current]:
                    if rel in causal_rels and related not in visited:
                        new_chain = chain + [f"{current} --[{rel}]--> {related}"]
                        new_evidence = evidence + [(current, rel, related, c)]
                        new_conf = conf * c
                        
                        if self._concepts_match(related, effect):
                            if new_conf > best_confidence:
                                best_chain = {
                                    'chain': new_chain,
                                    'evidence': new_evidence,
                                    'confidence': new_conf
                                }
                                best_confidence = new_conf
                        
                        visited.add(related)
                        queue.append((related, new_chain, new_evidence, new_conf, depth + 1))
        
        return best_chain
    
    def _extract_concepts(self, text: str) -> List[str]:
        """从文本中提取概念"""
        concepts = []
        
        for concept in self._concept_index.keys():
            if concept in text:
                concepts.append(concept)
        
        if not concepts:
            words = text.replace('为什么', '').replace('？', '').replace('会', '').split()
            concepts = [w for w in words if len(w) >= 2]
        
        return concepts
    
    def _synthesize_answer(self, question: str, derived_concepts: Set[str], 
                          reasoning_chain: List[str]) -> str:
        """综合推理链生成答案"""
        if not derived_concepts:
            return '无法通过概念图谱推理出答案'
        
        if '为什么' in question:
            core_concepts = list(derived_concepts)[:3]
            answer = f"因为{' → '.join(core_concepts)}"
            return answer
        
        return f"推理结果: {', '.join(list(derived_concepts)[:5])}"


class ReasoningChain:
    """推理链管理器"""
    
    def __init__(self):
        self.steps = []
        self.confidence = 1.0
    
    def add_step(self, premise: str, conclusion: str, rule: str, confidence: float):
        """添加推理步骤"""
        self.steps.append({
            'premise': premise,
            'conclusion': conclusion,
            'rule': rule,
            'confidence': confidence
        })
        self.confidence *= confidence
    
    def get_final_conclusion(self) -> str:
        """获取最终结论"""
        if self.steps:
            return self.steps[-1]['conclusion']
        return None
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'steps': self.steps,
            'confidence': self.confidence,
            'final_conclusion': self.get_final_conclusion()
        }


if __name__ == "__main__":
    print("=" * 60)
    print("深度推理引擎测试")
    print("=" * 60)
    
    test_triples = [
        ("人", "IS_A", "生物", 0.95),
        ("苏格拉底", "IS_A", "人", 0.99),
        ("生物", "都会", "死亡", 0.98),
        ("下雨", "CAUSES", "地面湿", 0.90),
        ("地面湿", "CAUSES", "摩擦力降低", 0.85),
        ("摩擦力降低", "CAUSES", "路面滑", 0.88),
        ("乌鸦", "HAS_PROPERTY", "黑色", 0.92),
        ("水", "HAS_PROPERTY", "流动性", 0.95),
        ("水", "HAS_PROPERTY", "可蒸发", 0.90),
        ("空气", "HAS_PROPERTY", "流动性", 0.93),
    ]
    
    engine = ReasoningEngine(concept_graph=test_triples)
    
    print("\n【测试1】演绎推理：苏格拉底会死")
    print("-" * 60)
    result = engine.deductive_reasoning(
        premises=["所有人都会死亡", "苏格拉底是人"],
        conclusion="苏格拉底会死亡"
    )
    print(f"推理有效: {result['valid']}")
    print(f"置信度: {result['confidence']:.2f}")
    print("推理链:")
    for step in result['reasoning_chain']:
        print(f"  {step}")
    
    print("\n【测试2】归纳推理：乌鸦都是黑色的")
    print("-" * 60)
    result = engine.inductive_reasoning([
        "乌鸦1是黑色的",
        "乌鸦2是黑色的",
        "乌鸦3是黑色的",
        "乌鸦4是黑色的"
    ])
    print(f"归纳结论: {result['generalization']}")
    print(f"置信度: {result['confidence']:.2f}")
    print(f"支持例子数: {result['support_count']}")
    
    print("\n【测试3】类比推理：空气像水")
    print("-" * 60)
    result = engine.analogical_reasoning(
        source={'entity': '水', 'attributes': ['流动性', '可蒸发', '无色']},
        target={'entity': '空气', 'attributes': ['流动性', '无色']}
    )
    print(f"推断属性: {result['inferred_attributes']}")
    print(f"相似度: {result['similarity']:.2f}")
    print(f"置信度: {result['confidence']:.2f}")
    print(f"推理说明: {result['reasoning']}")
    
    print("\n【测试4】因果推理：下雨导致地面湿")
    print("-" * 60)
    result = engine.causal_reasoning("下雨", "地面湿")
    print(f"存在因果关系: {result['is_causal']}")
    print(f"置信度: {result['confidence']:.2f}")
    print("因果链:")
    for chain in result['causal_chain']:
        print(f"  {chain}")
    
    print("\n【测试5】多步推理：为什么下雨后路面会滑？")
    print("-" * 60)
    result = engine.multi_step_reasoning("为什么下雨后路面会滑？", steps=5)
    print(f"答案: {result['answer']}")
    print(f"置信度: {result['confidence']:.2f}")
    print(f"使用步数: {result['steps_used']}")
    print("推理链:")
    for step in result['reasoning_chain']:
        print(f"  {step}")
    
    print("\n" + "=" * 60)
    print("所有测试完成")
    print("=" * 60)