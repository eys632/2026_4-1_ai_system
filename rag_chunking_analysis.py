#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG 청킹 및 벡터화 분석 도구
각 RAG 시스템의 청킹 방식과 벡터화 결과를 상세히 분석하고 저장합니다.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("🔍 RAG 시스템 청킹 & 벡터화 분석")
print("=" * 100)

# ============================================================================
# 1️⃣ 데이터 준비
# ============================================================================
print("\n📚 [1단계] 데이터 준비\n")

jju_ai_documents = [
    {"title": "학과소개", "content": "전주대학교 인공지능학과는 2021년도에 교육부 지정 첨단학과로 신설되었습니다. 전북권 유일의 인공지능 전문학과로서 인공지능에 관련된 전문적인 교육과 연구를 수행합니다. 창의적 융합 엔지니어 양성을 목표로 합니다.", "category": "학과안내"},
    {"title": "교육모토", "content": "AI를 직접 보고 느끼며 재미있고 쉽게 배운다. 학생들의 눈높이에 맞춰 인공지능을 가능한 쉽고 재미있게 가르칩니다.", "category": "학과안내"},
    {"title": "학과의 장점", "content": "교육부 지정 첨단학과로서 창의적이고 재미있는 취업 중심의 교육 프로그램 운영. 해외교육연수, 장학금, 개인 노트북/사물함 제공. 인공지능 관련 다양한 전문경력을 보유한 최고의 교수진.", "category": "학과안내"},
    {"title": "혁신융합대학사업", "content": "교육부 주관 첨단분야 혁신융합대학사업(AI분야)으로 선정. 매년 약 15억원 규모의 재정 지원. 표준화된 교육과정 개발, 산업계 협업, 국가 수준의 첨단분야 인재 양성.", "category": "학과안내"},
    {"title": "인공지능이 정말 배워야 하나요?", "content": "인공지능은 현재와 미래의 핵심기술이며, 산업과 사회 전반에 큰 변화를 가져올 것으로 예상됩니다. 모든 분야에서 필수적인 기술이 되고 있습니다.", "category": "신입생안내"},
    {"title": "컴퓨터공학과와의 차이점", "content": "컴퓨터공학과는 일반적인 프로그램 개발자를 양성하지만, 인공지능학과는 빅데이터 분석, 머신러닝, 딥러닝으로 대표되는 인공지능 전문 개발자를 양성합니다.", "category": "신입생안내"},
    {"title": "수업을 따라갈 수 있을까요?", "content": "저희 학과 교육모토는 AI를 직접 보고 느끼며 재미있고 쉽게 배운다는 것입니다. 고등학교 때까지 평범하게 공부한 학생들도 만족하며 공부하고 있습니다.", "category": "신입생안내"},
    {"title": "입학 요건은?", "content": "특정 학력수준이 정해져 있지 않습니다. 전주대학교 입학 수준이면 누구나 입학 가능합니다. 수학이나 프로그래밍을 사전에 준비하면 도움이 됩니다.", "category": "신입생안내"},
    {"title": "1학년 교과목", "content": "파이썬기초및실습, 알기쉬운확률통계, 논리적문제해결, 영상이해, 인공지능수학기초, 인공지능기초와활용", "category": "학사정보"},
    {"title": "2학년 교과목", "content": "AI알고리즘, 인공지능수학, 데이터분석기초, 프로그래밍기초와실습, 서비스러닝, 기계학습, 딥러닝, 문제해결과알고리즘, 리눅스운영체제", "category": "학사정보"},
    {"title": "3학년 교과목", "content": "데이터베이스, 로봇프로그래밍기초, 비지도학습, 유전알고리즘, 클라우드컴퓨팅, 강화학습, 첨단신경망, 자연어처리, 파이썬웹프로그래밍, AI리빙랩(1), IoT프로그래밍", "category": "학사정보"},
    {"title": "4학년 교과목", "content": "지식표현과추론, 인공지능시스템, 지능HCI, 음성인식, 인공지능세미나, AI리빙랩(2), 인공지능과윤리, 논문", "category": "학사정보"},
    {"title": "취업전망", "content": "인공지능 분야의 취업 전망은 매우 밝습니다. 인공지능은 다양한 분야에서 필수적인 역할을 수행하고 있으며, 산업 현장에서 좋은 대우와 많은 기회를 제공하고 있습니다.", "category": "진로취업"},
    {"title": "1기 졸업생", "content": "인공지능학과 1기 졸업생 전원 취업 및 대학원(서강대학교 등) 진학. 높은 취업률을 달성했습니다.", "category": "진로취업"},
    {"title": "2024년 CO-SHOW 수상", "content": "인공지능학과 1학년 재학생들이 첨단 분야 경진대회 2024년 CO-SHOW에서 한국연구재단 이사장상을 수상했습니다.", "category": "학과활동"},
    {"title": "2025 AI융합 문제발굴 산학연계 해커톤", "content": "한태희 외 타대학 학생들의 연합팀이 AI기반 카테고리 자동분류 모델학습 및 분석도구 개발 주제로 정확도 0.921을 달성하여 금상을 수상했습니다.", "category": "학과활동"},
]

print(f"✅ {len(jju_ai_documents)}개의 문서 준비 완료\n")

# ============================================================================
# 2️⃣ 청킹 분석 (Chunking)
# ============================================================================
print("=" * 100)
print("📝 [2단계] 청킹(Chunking) 분석")
print("=" * 100 + "\n")

chunking_analysis = {
    'Basic RAG': {'chunks': [], 'chunk_count': 0, 'avg_chunk_size': 0},
    'Improved RAG': {'chunks': [], 'chunk_count': 0, 'avg_chunk_size': 0},
    'LLM-Aware RAG': {'chunks': [], 'chunk_count': 0, 'avg_chunk_size': 0}
}

# 각 RAG별로 청킹 분석
for method in ['Basic RAG', 'Improved RAG', 'LLM-Aware RAG']:
    print(f"🔍 {method} - 청킹 분석")
    print("-" * 100)
    
    chunks = []
    for i, doc in enumerate(jju_ai_documents):
        # 간단한 청킹: 문장 기반
        sentences = doc['content'].split('. ')
        
        for j, sentence in enumerate(sentences):
            if sentence.strip():
                chunk = {
                    'doc_id': i,
                    'doc_title': doc['title'],
                    'doc_category': doc['category'],
                    'chunk_id': len(chunks),
                    'chunk_text': sentence.strip(),
                    'chunk_size': len(sentence.strip()),
                    'char_length': len(sentence.strip()),
                    'word_count': len(sentence.strip().split())
                }
                chunks.append(chunk)
    
    chunking_analysis[method]['chunks'] = chunks
    chunking_analysis[method]['chunk_count'] = len(chunks)
    chunking_analysis[method]['avg_chunk_size'] = np.mean([c['char_length'] for c in chunks])
    
    print(f"   • 총 청크 수: {len(chunks)}")
    print(f"   • 평균 청크 크기: {chunking_analysis[method]['avg_chunk_size']:.1f} 글자")
    print(f"   • 최소/최대 크기: {min([c['char_length'] for c in chunks])}/{max([c['char_length'] for c in chunks])} 글자\n")

# 청킹 결과 저장
os.makedirs('rag_analysis', exist_ok=True)

for method, data in chunking_analysis.items():
    with open(f'rag_analysis/{method.lower().replace(" ", "_")}_chunks.json', 'w', encoding='utf-8') as f:
        json.dump(data['chunks'][:5], f, ensure_ascii=False, indent=2)  # 처음 5개만
    print(f"✅ {method} 청킹 결과 저장: rag_analysis/{method.lower().replace(' ', '_')}_chunks.json")

print("\n" + "=" * 100)

# ============================================================================
# 3️⃣ 벡터화 분석 (Vectorization)
# ============================================================================
print("🔢 [3단계] 벡터화(Vectorization) 분석")
print("=" * 100 + "\n")

documents_text = [doc['content'] for doc in jju_ai_documents]

vectorization_analysis = {
    'Basic RAG': {'vectorizer': None, 'embeddings': None, 'feature_names': [], 'vocab_size': 0},
    'Improved RAG': {'vectorizer': None, 'embeddings': None, 'feature_names': [], 'vocab_size': 0},
    'LLM-Aware RAG': {'vectorizer': None, 'embeddings': None, 'feature_names': [], 'vocab_size': 0}
}

# Basic RAG: 100개 특성
print("🔍 Basic RAG - Vectorization")
print("-" * 100)
vectorizer_basic = TfidfVectorizer(max_features=100)
embeddings_basic = vectorizer_basic.fit_transform(documents_text)
vectorization_analysis['Basic RAG']['vectorizer'] = vectorizer_basic
vectorization_analysis['Basic RAG']['embeddings'] = embeddings_basic
vectorization_analysis['Basic RAG']['feature_names'] = vectorizer_basic.get_feature_names_out()
vectorization_analysis['Basic RAG']['vocab_size'] = len(vectorizer_basic.get_feature_names_out())
print(f"   • Vectorizer: TfidfVectorizer")
print(f"   • Max Features: 100")
print(f"   • 실제 특성 수: {vectorization_analysis['Basic RAG']['vocab_size']}")
print(f"   • 임베딩 행렬 크기: {embeddings_basic.shape}")
print(f"   • 상위 10개 특성: {', '.join(vectorization_analysis['Basic RAG']['feature_names'][:10])}\n")

# Improved RAG: 200개 특성 + Bigram
print("🔍 Improved RAG - Vectorization")
print("-" * 100)
vectorizer_improved = TfidfVectorizer(max_features=200, ngram_range=(1,2))
embeddings_improved = vectorizer_improved.fit_transform(documents_text)
vectorization_analysis['Improved RAG']['vectorizer'] = vectorizer_improved
vectorization_analysis['Improved RAG']['embeddings'] = embeddings_improved
vectorization_analysis['Improved RAG']['feature_names'] = vectorizer_improved.get_feature_names_out()
vectorization_analysis['Improved RAG']['vocab_size'] = len(vectorizer_improved.get_feature_names_out())
print(f"   • Vectorizer: TfidfVectorizer with Bigram")
print(f"   • Max Features: 200")
print(f"   • NGram Range: (1, 2)")
print(f"   • 실제 특성 수: {vectorization_analysis['Improved RAG']['vocab_size']}")
print(f"   • 임베딩 행렬 크기: {embeddings_improved.shape}")
print(f"   • 상위 10개 특성: {', '.join(vectorization_analysis['Improved RAG']['feature_names'][:10])}\n")

# LLM-Aware RAG: 200개 특성 + Bigram
print("🔍 LLM-Aware RAG - Vectorization")
print("-" * 100)
vectorizer_llm = TfidfVectorizer(max_features=200, ngram_range=(1,2))
embeddings_llm = vectorizer_llm.fit_transform(documents_text)
vectorization_analysis['LLM-Aware RAG']['vectorizer'] = vectorizer_llm
vectorization_analysis['LLM-Aware RAG']['embeddings'] = embeddings_llm
vectorization_analysis['LLM-Aware RAG']['feature_names'] = vectorizer_llm.get_feature_names_out()
vectorization_analysis['LLM-Aware RAG']['vocab_size'] = len(vectorizer_llm.get_feature_names_out())
print(f"   • Vectorizer: TfidfVectorizer with Bigram")
print(f"   • Max Features: 200")
print(f"   • NGram Range: (1, 2)")
print(f"   • 실제 특성 수: {vectorization_analysis['LLM-Aware RAG']['vocab_size']}")
print(f"   • 임베딩 행렬 크기: {embeddings_llm.shape}")
print(f"   • 상위 10개 특성: {', '.join(vectorization_analysis['LLM-Aware RAG']['feature_names'][:10])}\n")

# 벡터화 결과 저장
for method in ['Basic RAG', 'Improved RAG', 'LLM-Aware RAG']:
    method_key = method.lower().replace(' ', '_')
    
    # 특성명 저장
    with open(f'rag_analysis/{method_key}_feature_names.json', 'w', encoding='utf-8') as f:
        json.dump(vectorization_analysis[method]['feature_names'].tolist()[:20], f, ensure_ascii=False, indent=2)

print("=" * 100)

# ============================================================================
# 4️⃣ 상세 분석 요약
# ============================================================================
print("\n📊 [4단계] 분석 요약\n")

summary_table = []
for method in ['Basic RAG', 'Improved RAG', 'LLM-Aware RAG']:
    summary_table.append({
        'RAG 방식': method,
        '청크 수': chunking_analysis[method]['chunk_count'],
        '평균 청크크기': f"{chunking_analysis[method]['avg_chunk_size']:.0f}",
        '특성 수': vectorization_analysis[method]['vocab_size'],
        '임베딩 크기': f"{vectorization_analysis[method]['embeddings'].shape}"
    })

df = pd.DataFrame(summary_table)
print(df.to_string(index=False))

# 저장
df.to_csv('rag_analysis/chunking_vectorization_summary.csv', encoding='utf-8-sig', index=False)
print("\n✅ 분석 요약 저장: rag_analysis/chunking_vectorization_summary.csv")

# ============================================================================
# 5️⃣ 문서별 벡터 통계
# ============================================================================
print("\n📈 문서별 벡터 통계\n")

doc_stats = []
for i, doc in enumerate(jju_ai_documents):
    basic_vec = embeddings_basic[i].toarray()[0]
    improved_vec = embeddings_improved[i].toarray()[0]
    llm_vec = embeddings_llm[i].toarray()[0]
    
    doc_stats.append({
        '문서ID': i,
        '제목': doc['title'][:20],
        '카테고리': doc['category'],
        'Basic_비영(%)': f"{(basic_vec != 0).sum() / len(basic_vec) * 100:.1f}",
        'Improved_비영(%)': f"{(improved_vec != 0).sum() / len(improved_vec) * 100:.1f}",
        'LLM_비영(%)': f"{(llm_vec != 0).sum() / len(llm_vec) * 100:.1f}"
    })

df_stats = pd.DataFrame(doc_stats)
print(df_stats.to_string(index=False))
df_stats.to_csv('rag_analysis/document_vector_statistics.csv', encoding='utf-8-sig', index=False)
print("\n✅ 문서별 벡터 통계 저장: rag_analysis/document_vector_statistics.csv")

print("\n" + "=" * 100)
print("✅ 모든 분석 완료!")
print("=" * 100)
print("\n📁 생성된 파일:")
print("   • rag_analysis/basic_rag_chunks.json")
print("   • rag_analysis/improved_rag_chunks.json")
print("   • rag_analysis/llm_aware_rag_chunks.json")
print("   • rag_analysis/basic_rag_feature_names.json")
print("   • rag_analysis/improved_rag_feature_names.json")
print("   • rag_analysis/llm_aware_rag_feature_names.json")
print("   • rag_analysis/chunking_vectorization_summary.csv")
print("   • rag_analysis/document_vector_statistics.csv")
print("\n" + "=" * 100)
