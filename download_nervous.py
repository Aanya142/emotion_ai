from icrawler.builtin import BingImageCrawler

crawler = BingImageCrawler(
    storage={'root_dir': 'dataset/train/nervous'}
)

keywords = [
    "nervous face",
    "anxious expression",
    "worried face",
    "stressed person face"
]

for kw in keywords:
    crawler.crawl(
        keyword=kw,
        max_num=100
    )
