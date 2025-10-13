"""
Test script cho Vietnamese News Pipeline
"""

import json
from pathlib import Path
from Vietnamese_Components.vietnamese_news_pipeline import VietnameseNewsPipeline

# Example data giống format của bạn
EXAMPLE_NEWS = {
    "single_documents": [
        {
            "title": "Máy bay chở 180 hành khách phải 'quay đầu' vì động cơ bốc cháy",
            "anchor_text": "Động cơ của một chiếc máy bay từ Mexico đến Los Angeles (Mỹ) đã bất ngờ bốc cháy khi đang ở trên không, khiến phi công phải 'quay đầu' và hạ cánh khẩn cấp.",
            "raw_text": """Chuyến bay của hãng Viva Aerobus mang số hiệu VB518, có sức chứa 186 hành khách, khởi hành từ Guadalajara, Mexico, tối 24/8 (giờ địa phương).
Chuyến bay dự kiến kéo dài 3 tiếng, tuy nhiên, khoảng 10 phút sau khi cất cánh, hành khách phát hiện tia lửa bắn ra từ động cơ bên phải của máy bay. Một số hành khách cho biết đã nghe thấy "âm thanh của một vụ nổ".
Tình huống này đã khiến nhiều hàng khách hoảng loạn, một số người than khóc, la hét và cầu nguyện.
Phi hành đoàn nhận được cảnh báo khi máy bay đang ở độ cao tương đương 4.000 m và đã quay trở lại hạ cánh tại sân bay Guadalajara, 45 phút sau khi cất cánh.
Rất may mắn, vụ việc này không biến thành một thảm họa hàng không.
Viva Aerobus cho biết tia lửa điện trong động cơ là do "ma sát của kim loại".
Theo hãng hàng không này, các hành khách trên chuyến bay đã được đưa đến một khách sạn và tiếp tục hành trình vào sáng hôm sau.
Nguyên nhân vụ việc đang tiếp tục được điều tra làm rõ."""
        },
        {
            "title": "Mexico: Máy bay hạ cánh khẩn cấp vì động cơ 'phun lửa'",
            "anchor_text": "Một chiếc máy bay của hãng hàng không giá rẻ Viva Aerobus từ Guadalajara (Mexico) đến Los Angeles (Mỹ) đã phải quay đầu và hạ cánh khẩn cấp do bị cháy động cơ.",
            "raw_text": """Theo RT, vài phút sau khi chiếc máy bay Airbus A320 cất cánh lúc 22h ngày 23/8, một tiếng nổ lớn đã vang lên và lửa bắt đầu phụt ra từ động cơ bên phải.
Các hành khách vội vàng báo cho phi hành đoàn, và máy bay lập tức quay đầu, hạ cánh an toàn ở Guadalajara.
Đoạn video ghi lại vụ việc cho thấy sự cố xảy ra khi máy bay đang bay ở độ cao khá thấp trên khu vực đông dân cư. Vụ cháy khiến các hành khách hoảng loạn, khóc lóc, la hét và cầu nguyện.
Không ai trong số các thành viên phi hành đoàn hoặc 186 hành khách bị thương.
Hãng hàng không Viva Aerobus đã sắp xếp một chuyến bay khác đến Los Angeles vào sáng 24/8."""
        }
    ],
    "summary": "Chuyến bay của hãng Viva Aerobus mang số hiệu VB518, có sức chứa 186 hành khách, khởi hành từ Guadalajara, Mexico, tối 24/8 (giờ địa phương), dự kiến kéo dài 3 tiếng. Khoảng 10 phút sau khi cất cánh, hành khách phát hiện tia lửa bắn ra từ động cơ bên phải của máy bay. 45 phút sau khi cất cánh, khi máy bay đang ở độ cao tương đương 4.000 m, phi hành đoàn đã cho máy bay quay trở lại hạ cánh tại sân bay Guadalajara. Rất may mắn, vụ việc này không biến thành một thảm họa hàng không. Nguyên nhân vụ việc đang tiếp tục được điều tra làm rõ.",
    "category": "Khoa học - Công nghệ"
}


def create_test_jsonl():
    """Create test JSONL file"""
    test_path = Path('test_data')
    test_path.mkdir(exist_ok=True)
    
    jsonl_file = test_path / 'test_news.jsonl'
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(EXAMPLE_NEWS, ensure_ascii=False) + '\n')
    
    print(f"Created test file: {jsonl_file}")
    return jsonl_file


def test_components():
    """Test từng component riêng lẻ"""
    print("="*60)
    print("TESTING VIETNAMESE COMPONENTS")
    print("="*60)
    
    from vietnamese_model_replacements import VietnameseModelManager
    
    manager = VietnameseModelManager(device="cuda")
    
    # Prepare test text
    text = EXAMPLE_NEWS['single_documents'][0]['raw_text']
    
    # Test 1: Summarizer
    print("\n[1/4] Testing Summarizer...")
    manager.load_summarizer()
    summary = manager.summarizer.generate_summary(text)
    print(f"✓ Generated summary:")
    print(f"  {summary}")
    
    # Test 2: Decomposer
    print("\n[2/4] Testing Atomic Fact Decomposer...")
    manager.load_decomposer()
    facts = manager.decomposer.decompose(summary)
    print(f"✓ Extracted {len(facts)} atomic facts:")
    for i, fact in enumerate(facts, 1):
        print(f"  {i}. {fact}")
    
    # Test 3: KG Extractor
    print("\n[3/4] Testing Knowledge Graph Extractor...")
    manager.load_kg_extractor()
    entities, relations = manager.kg_extractor.extract_named_entities_and_relations(text)
    print(f"✓ Extracted {len(entities)} entities:")
    print(f"  {entities[:10]}")
    print(f"✓ Extracted {len(relations)} relations:")
    for i, (subj, pred, obj) in enumerate(relations[:5], 1):
        print(f"  {i}. ({subj}) --[{pred}]--> ({obj})")
    
    # Test 4: Fact Verifier
    print("\n[4/4] Testing Fact Verifier...")
    manager.load_verifier()
    
    test_facts = [
        "Chuyến bay VB518 khởi hành từ Guadalajara",  # True
        "Máy bay bị cháy động cơ sau khi cất cánh",    # True
        "Có 200 hành khách trên máy bay",               # False
    ]
    
    for fact in test_facts:
        score, feedback = manager.verifier.verify_fact(text, fact)
        status = "✓" if score > 0.5 else "✗"
        print(f"  {status} Fact: {fact}")
        print(f"     Score: {score:.2f}, Feedback: {feedback}")
    
    print("\n" + "="*60)


def test_full_pipeline():
    """Test full pipeline"""
    print("\n" + "="*60)
    print("TESTING FULL PIPELINE")
    print("="*60)
    
    # Create test file
    test_file = create_test_jsonl()
    
    # Initialize pipeline
    pipeline = VietnameseNewsPipeline(device="cuda")
    
    # Process
    results = pipeline.process_jsonl_file(
        jsonl_path=str(test_file),
        output_dir='./test_output_vietnamese',
        num_iterations=2,
        max_items=1
    )
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    result = results[0]
    print(f"\nNews ID: {result['id']}")
    print(f"Category: {result['category']}")
    print(f"\nReference Summary:")
    print(f"  {result['reference_summary']}")
    print(f"\nInitial Summary:")
    print(f"  {result['initial_summary']}")
    print(f"\nFinal Summary:")
    print(f"  {result['final_summary']}")
    print(f"\nScore Progression:")
    for iter_data in result['iteration_history']:
        print(f"  Iteration {iter_data['iteration']}: {iter_data['score']:.4f}")
    
    improvement = result['iteration_history'][-1]['score'] - result['iteration_history'][0]['score']
    print(f"\nImprovement: {improvement:+.4f}")
    print("="*60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='full',
                        choices=['components', 'full', 'both'])
    args = parser.parse_args()
    
    try:
        if args.mode in ['components', 'both']:
            test_components()
        
        if args.mode in ['full', 'both']:
            test_full_pipeline()
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()