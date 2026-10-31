"""

Nguon: https://www.pharmacity.vn/cham-soc-ca-nhan  (goi API JSON public cua trang)

Chay:  scrapy runspider step1_pharmacity_scrapy.py
Xuat:  pharmacity_personal_care.csv  va  pharmacity_personal_care.json
"""
import scrapy

API_URL = "https://api-gateway.pharmacity.vn/pmc-ecm-product/api/public/search/index"
PAGE_LIMIT = 50
TARGET_RECORDS = 300

CATEGORY_SLUGS = [
    "san-pham-khu-mui", "san-pham-phong-tam", "cham-soc-toc",
    "cham-soc-rang-mieng", "ve-sinh-phu-nu", "cham-soc-nam-gioi", "cham-soc-co-the",
]

FIELDS = [
    "ProductID", "ProductName", "Category", "CategoryCode", "SourceSlug",
    "Brand", "Price", "OriginalPrice", "DiscountPercent", "Unit",
    "Rating", "TotalSold", "URL",
]


class PharmacitySpider(scrapy.Spider):
    name = "pharmacity"
    api_url = API_URL  # co the ghi de: -a api_url=http://...

    custom_settings = {
        "ROBOTSTXT_OBEY": False,          # API gateway, khong phai trang HTML
        "DOWNLOAD_DELAY": 1.0,
        "CONCURRENT_REQUESTS": 4,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 2,
        "AUTOTHROTTLE_ENABLED": True,
        "RETRY_TIMES": 3,
        "DOWNLOAD_TIMEOUT": 15,
        "LOG_LEVEL": "INFO",
        "DEFAULT_REQUEST_HEADERS": {
            "Accept": "application/json",
            "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
        },
        "FEED_EXPORT_ENCODING": "utf-8-sig",
        "FEED_EXPORT_FIELDS": FIELDS,
        "FEEDS": {
            "pharmacity_personal_care.csv": {"format": "csv", "overwrite": True},
            "pharmacity_personal_care.json": {
                "format": "json", "overwrite": True, "indent": 2,
                "encoding": "utf-8", "ensure_ascii": False,
            },
        },
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen_skus = set()  # chong trung giua cac danh muc

    # ---- tao request ----
    def build_url(self, slug, index):
        return (
            f"{self.api_url}?platform=1&index={index}&limit={PAGE_LIMIT}&total=0"
            f"&refresh=true&page=category&page_slug={slug}"
        )

    async def start(self):  # Scrapy >= 2.13
        for slug in CATEGORY_SLUGS:
            yield self.page_request(slug, 1)

    def start_requests(self):  # Scrapy < 2.13
        for slug in CATEGORY_SLUGS:
            yield self.page_request(slug, 1)

    def page_request(self, slug, index):
        return scrapy.Request(
            self.build_url(slug, index),
            callback=self.parse_page,
            cb_kwargs={"slug": slug, "index": index},
            errback=self.on_error,
        )

    # ---- xu ly 1 trang ----
    def parse_page(self, response, slug, index):
        try:
            data = (response.json().get("data")) or {}
        except ValueError:
            self.logger.warning(f"[{slug}] index={index}: phan hoi khong phai JSON")
            return

        items = data.get("items") or []
        if not items:
            self.logger.info(f"[{slug}] het du lieu o index={index}")
            return

        new_count = 0
        for item in items:
            sku = item.get("sku")
            if not sku or sku in self.seen_skus:
                continue
            self.seen_skus.add(sku)
            new_count += 1
            yield self.flatten_product(item, slug)

        self.logger.info(f"[{slug}] index={index}: +{new_count} san pham moi")

        try:
            total = int(data.get("total", 0))
        except (TypeError, ValueError):
            total = 0

        # Phan trang: con trang tiep va trang nay co du lieu moi
        if index * PAGE_LIMIT < total and new_count > 0:
            yield self.page_request(slug, index + 1)

    @staticmethod
    def flatten_product(item, source_slug):
        variant = (item.get("variants") or [{}])[0]
        sub = item.get("sub_data") or {}
        slug = item.get("slug")
        return {
            "ProductID": item.get("sku"),
            "ProductName": item.get("name"),
            "Category": item.get("category_name"),
            "CategoryCode": item.get("category_code"),
            "SourceSlug": source_slug,
            "Brand": item.get("brand_name"),
            "Price": variant.get("price"),
            "OriginalPrice": variant.get("original_price"),
            "DiscountPercent": variant.get("discount_percent"),
            "Unit": variant.get("unit_name"),
            "Rating": sub.get("score"),
            "TotalSold": sub.get("total_sold"),
            "URL": f"https://www.pharmacity.vn/{slug}" if slug else None,
        }

    def on_error(self, failure):
        self.logger.warning(f"Request loi: {failure.request.url} -> {failure.value}")

    def closed(self, reason):
        n = len(self.seen_skus)
        self.logger.info(f"DONE ({reason}). Thu duoc {n} san pham duy nhat.")
        if n < TARGET_RECORDS:
            self.logger.warning(f"Chi co {n} < {TARGET_RECORDS} ban ghi, kiem tra lai CATEGORY_SLUGS.")