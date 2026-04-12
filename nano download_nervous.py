from icrawler.builtin import GoogleImageCrawler

crawler = GoogleImageCrawler(
    storage={'root_dir': 'dataset/train/nervous'}
)

crawler.crawl(
    keyword='nervous face expression',
    max_num=300