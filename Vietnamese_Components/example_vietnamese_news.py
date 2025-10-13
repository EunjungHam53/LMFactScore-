"""
Example script cho Vietnamese News Summarization
Test với data format của bạn
"""

import json
from vietnamese_news_pipeline import VietnameseNewsPipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Example data (định dạng giống của bạn)
EXAMPLE_DATA = {
    "single_documents": [
        {
            "title": "Máy bay chở 180 hành khách phải 'quay đầu' vì động cơ bốc cháy",
            "anchor_text": "Động cơ của một chiếc máy bay từ Mexico đến Los Angeles (Mỹ) đã bất ngờ bốc cháy khi đang ở trên không, khiến phi công phải 'quay đầu' và hạ cánh khẩn cấp.",
            "raw_text": "Chuyến bay của hãng Viva Aerobus mang số hiệu VB518, có sức chứa 186 hành khách, khởi hành từ Guadalajara, Mexico, tối 24/8 (giờ địa phương).\nChuyến bay dự kiến kéo dài 3 tiếng, tuy nhiên, khoảng 10 phút sau khi cất cánh, hành khách phát hiện tia lửa bắn ra từ động cơ bên phải của máy bay. Một số hành khách cho biết đã nghe thấy 'âm thanh của một vụ nổ'.\nTình huống này đã khiến nhiều hàng khách hoảng loạn, một số người than khóc, la hét và cầu nguyện.\nPhi hành đoàn nhận được cảnh báo khi máy bay đang ở độ cao tương đương 4.000 m và đã quay trở lại hạ cánh tại sân bay Guadalajara, 45 phút sau khi cất cánh.\nRất may mắn, vụ việc này không biến thành một thảm họa hàng không.\nViva Aerobus cho biết tia lửa điện trong động cơ là do 'ma sát của kim loại'.\nTheo hãng hàng không này, các hành khách trên chuyến bay đã được đưa đến một khách sạn và tiếp tục hành trình vào sáng hôm sau.\nNguyên nhân vụ việc đang tiếp tục được điều tra làm rõ.\nĐây là lần thứ hai một chiếc máy bay của Viva Aerobus phải hạ cánh khẩn cấp ở Mexico trong 3 tháng qua.\nNgày 22/5, một chuyến bay khởi hành từ Villahermosa và hướng đến Mexico City thì một chú chim bất ngờ lao vào tuabin động cơ, khiến máy bay phải hạ cánh khẩn cấp, may mắn không xảy ra tai nạn đáng tiếc nào."
        },
        {
            "title": "Mexico: Máy bay hạ cánh khẩn cấp vì động cơ 'phun lửa'",
            "anchor_text": "Một chiếc máy bay của hãng hàng không giá rẻ Viva Aerobus từ Guadalajara (Mexico) đến Los Angeles (Mỹ) đã phải quay đầu và hạ cánh khẩn cấp do bị cháy động cơ.",
            "raw_text": "Theo RT, vài phút sau khi chiếc máy bay Airbus A320 cất cánh lúc 22h ngày 23/8, một tiếng nổ lớn đã vang lên và lửa bắt đầu phụt ra từ động cơ bên phải.\nCác hành khách vội vàng báo cho phi hành đoàn, và máy bay lập tức quay đầu, hạ cánh an toàn ở Guadalajara.\nĐoạn video ghi lại vụ việc cho thấy sự cố xảy ra khi máy bay đang bay ở độ cao khá thấp trên khu vực đông dân cư. Vụ cháy khiến các hành khách hoảng loạn, khóc lóc, la hét và cầu nguyện.\nKhông ai trong số các thành viên phi hành đoàn hoặc 186 hành khách bị thương.\nHãng hàng không Viva Aerobus đã sắp xếp một chuyến bay khác đến Los Angeles vào sáng 24/8."
        }
    ],
    "summary": "Chuyến bay của hãng Viva Aerobus mang số hiệu VB518, có sức chứa 186 hành khách, khởi hành từ Guadalajara, Mexico, tối 24/8 (giờ địa phương), dự kiến kéo dài 3 tiếng. Khoảng 10 phút sau khi cất cánh, hành khách phát hiện tia lửa bắn ra từ động cơ bên phải của máy bay. 45 phút sau khi cất cánh, khi máy bay đang ở độ cao tương đương 4.000 m, phi hành đoàn đã cho máy bay quay trở lại hạ cánh tại sân bay Guadalajara. Rất may mắn, vụ việc này không biến thành một thảm họa hàng không. Nguyên nhân vụ việc đang tiếp tục được điều tra làm rõ.",
    "category": "Khoa học - Công nghệ"
}


def test_vietnamese_summarizer():
    """
    Test Vietnamese summarizer với example data
    """
    print("="*60)
    print("TESTING VIETNAMESE NEWS SUMMARIZATION")
    print("="*60)
    
    from vietnamese_news_pipeline import VietnameseNewsSummarizer
    
    # Test different models
    models = [
        ("VietAI/vit5-base", "Vietnamese T5 (fast)"),
        # ("VietAI/vit5-large", "Vietnamese T5 (better quality)"),
        # ("google/mt5-base", "Multilingual T5"),
    ]
    
    for model_name, description in models:
        print(f"\n[Testing {description}]")
        print(f"Model: {model_name}")
        
        try:
            summarizer = VietnameseNewsSummarizer(
                model_name=model_name,
                device="cuda"  # Change to "cpu" if no GPU
            )
            
            # Combine documents
            combined_text = ""
            for doc in EXAMPLE_DATA['single_documents']:
                combined_text += doc['title'] + "\n" + doc['raw_text'] + "\n\n"
            
            # Generate summary
            summary = summarizer.summarize(combined_text)
            
            print(f"\n✓ Generated Summary:")
            print(f"  {summary}")
            print(f"\n  Ground Truth:")
            print(f"  {EXAMPLE_DATA['summary']}")
            print(f"\n  Length: Generated={len(summary)} chars, GT={len(EXAMPLE_DATA['summary'])} chars")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()


def test_vietnamese_fact_checker():
    """
    Test Vietnamese fact checker
    """
    print("\n" + "="*60)
    print("TESTING VIETNAMESE FACT CHECKER")
    print("="*60)
    
    from vietnamese_news_pipeline import VietnameseFactChecker
    
    fact_checker = VietnameseFactChecker(
        model_name="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        device="cuda"
    )
    
    # Source text
    source = EXAMPLE_DATA['single_documents'][0]['raw_text']
    
    # Test cases: (statement, expected_result)
    test_cases = [
        ("Chuyến bay VB518 khởi hành từ Guadalajara", True),
        ("Máy bay có 186 hành khách", True),
        ("Động cơ bốc cháy sau 10 phút cất cánh", True),
        ("Máy bay đã rơi và gây thảm họa", False),  # Sai
        ("Tất cả hành khách đã thiệt mạng", False),  # Sai
    ]
    
    print("\nChecking facts against source document:")
    for statement, expected in test_cases:
        score, feedback = fact_checker.check_fact(source, statement)
        status = "✓" if (score > 0.5) == expected else "✗"
        
        print(f"\n{status} Statement: {statement}")
        print(f"   Expected: {'Supported' if expected else 'Not supported'}")
        print(f"   Score: {score:.2f}")
        if feedback:
            print(f"   Feedback: {feedback}")


def test_full_pipeline():
    """
    Test full pipeline với example data
    """
    print("\n" + "="*60)
    print("TESTING FULL VIETNAMESE NEWS PIPELINE")
    print("="*60)
    
    # Initialize pipeline
    pipeline = VietnameseNewsPipeline(
        summarizer_model="VietAI/vit5-base",
        fact_checker_model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        device="cuda"
    )
    
    # Process example item
    result = pipeline.process_single_item(EXAMPLE_DATA)
    
    # Print results
    print(f"\nCategory: {result['category']}")
    print(f"Number of documents: {result['num_documents']}")
    print(f"\nGenerated Summary:")
    print(f"  {result['generated_summary']}")
    print(f"\nGround Truth:")
    print(f"  {result['ground_truth_summary']}")
    
    if result['factuality_score'] is not None:
        print(f"\nFactuality Score: {result['factuality_score']:.2f}")


def test_jsonl_processing():
    """
    Test processing JSONL file
    """
    print("\n" + "="*60)
    print("TESTING JSONL FILE PROCESSING")
    print("="*60)
    
    # Create test JSONL file
    test_input = "test_input.jsonl"
    test_output = "test_output.jsonl"
    
    # Write example data to JSONL
    with open(test_input, 'w', encoding='utf-8') as f:
        f.write(json.dumps(EXAMPLE_DATA, ensure_ascii=False) + '\n')
        # You can add more items here
    
    print(f"\nCreated test file: {test_input}")
    
    # Initialize pipeline
    pipeline = VietnameseNewsPipeline(
        summarizer_model="VietAI/vit5-base",
        device="cuda"
    )
    
    # Process file
    pipeline.process_jsonl_file(
        input_path=test_input,
        output_path=test_output,
        max_items=5  # Process only first 5 items
    )
    
    print(f"\nResults saved to: {test_output}")
    
    # Read and display results
    with open(test_output, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            result = json.loads(line)
            print(f"\nItem {i}:")
            print(f"  Category: {result['category']}")
            print(f"  Summary: {result['generated_summary'][:100]}...")


def compare_models():
    """
    So sánh các models khác nhau
    """
    print("\n" + "="*60)
    print("COMPARING DIFFERENT MODELS")
    print("="*60)
    
    models = [
        ("VietAI/vit5-base", "Vietnamese T5 Base"),
        # Uncomment để test các models khác (cần nhiều VRAM hơn)
        # ("VietAI/vit5-large", "Vietnamese T5 Large"),
        # ("google/mt5-base", "Multilingual T5 Base"),
    ]
    
    # Combine documents
    combined_text = ""
    for doc in EXAMPLE_DATA['single_documents']:
        combined_text += doc['title'] + "\n" + doc['raw_text'] + "\n\n"
    
    results = []
    
    for model_name, description in models:
        print(f"\n[{description}]")
        try:
            from vietnamese_news_pipeline import VietnameseNewsSummarizer
            import time
            
            summarizer = VietnameseNewsSummarizer(
                model_name=model_name,
                device="cuda"
            )
            
            start_time = time.time()
            summary = summarizer.summarize(combined_text)
            elapsed = time.time() - start_time
            
            print(f"✓ Time: {elapsed:.2f}s")
            print(f"  Summary: {summary[:150]}...")
            
            results.append({
                'model': description,
                'time': elapsed,
                'summary': summary,
                'length': len(summary)
            })
            
        except Exception as e:
            print(f"✗ Error: {e}")
    
    # Compare results
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    for r in results:
        print(f"\n{r['model']}:")
        print(f"  Time: {r['time']:.2f}s")
        print(f"  Length: {r['length']} chars")


def create_sample_jsonl():
    """
    Tạo file JSONL mẫu với format đúng
    """
    print("\n" + "="*60)
    print("CREATING SAMPLE JSONL FILE")
    print("="*60)
    
    sample_file = "sample_vietnamese_news.jsonl"
    
    # Example data từ bạn
    sample_items = [
        EXAMPLE_DATA,
        # Có thể thêm nhiều items khác ở đây
    ]
    
    with open(sample_file, 'w', encoding='utf-8') as f:
        for item in sample_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✓ Created sample file: {sample_file}")
    print(f"  Format: JSONL (one JSON object per line)")
    print(f"  Items: {len(sample_items)}")
    print(f"\nYou can now run:")
    print(f"  python vietnamese_news_pipeline.py \\")
    print(f"    --input {sample_file} \\")
    print(f"    --output results.jsonl \\")
    print(f"    --device cuda")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Vietnamese News Pipeline')
    parser.add_argument(
        '--test',
        type=str,
        default='all',
        choices=['all', 'summarizer', 'fact_checker', 'pipeline', 'jsonl', 'compare', 'sample'],
        help='Which test to run'
    )
    
    args = parser.parse_args()
    
    try:
        if args.test in ['all', 'summarizer']:
            test_vietnamese_summarizer()
        
        if args.test in ['all', 'fact_checker']:
            test_vietnamese_fact_checker()
        
        if args.test in ['all', 'pipeline']:
            test_full_pipeline()
        
        if args.test in ['all', 'jsonl']:
            test_jsonl_processing()
        
        if args.test == 'compare':
            compare_models()
        
        if args.test == 'sample':
            create_sample_jsonl()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETED")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()