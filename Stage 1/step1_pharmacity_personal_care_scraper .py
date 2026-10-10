import json
import logging
import random
import time

import pandas as pd
import requests

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
API_URL = "https://api-gateway.pharmacity.vn/pmc-ecm-product/api/public/search/index"

# The 7 top-level subcategories under "Chăm sóc cá nhân"
# (https://www.pharmacity.vn/cham-soc-ca-nhan)
CATEGORY_SLUGS = [
    "san-pham-khu-mui",     # Sản phẩm khử mùi
    "san-pham-phong-tam",   # Sản phẩm phòng tắm
    "cham-soc-toc",         # Chăm sóc tóc
    "cham-soc-rang-mieng",  # Chăm sóc răng miệng
    "ve-sinh-phu-nu",       # Vệ sinh phụ nữ
    "cham-soc-nam-gioi",    # Chăm sóc nam giới
    "cham-soc-co-the",      # Chăm sóc cơ thể
]

PAGE_LIMIT = 50
REQUEST_DELAY_RANGE = (0.6, 1.4)
TIMEOUT = 15
TARGET_RECORDS = 300

OUTPUT_CSV = "pharmacity_personal_care.csv"
OUTPUT_JSON = "pharmacity_personal_care.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------
def polite_sleep():
    time.sleep(random.uniform(*REQUEST_DELAY_RANGE))


def fetch_page(page_slug, index):
    """Call the JSON API for one page of one category. Returns dict or None."""
    params = {
        "platform": 1,
        "index": index,
        "limit": PAGE_LIMIT,
        "total": 0,
        "refresh": "true",
        "page": "category",
        "page_slug": page_slug,
    }
    try:
        resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        log.warning(f"Request failed for {page_slug} index={index}: {e}")
        return None
    except json.JSONDecodeError:
        log.warning(f"Non-JSON response for {page_slug} index={index}")
        return None


def flatten_product(item, source_slug):
    """Turn one nested product dict from the API into a flat row."""
    variant = (item.get("variants") or [{}])[0]
    sub_data = item.get("sub_data") or {}
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
        "Rating": sub_data.get("score"),
        "TotalSold": sub_data.get("total_sold"),
        "URL": f"https://www.pharmacity.vn/{slug}" if slug else None,
    }


def crawl_category(page_slug, seen_skus, collected):
    """Page through one category slug until the API's reported total is reached."""
    index = 1
    total = None

    while True:
        log.info(f"[{page_slug}] fetching page index={index} ...")
        payload = fetch_page(page_slug, index)

        if payload is None:
            break

        data = payload.get("data") or {}
        items = data.get("items") or []

        if total is None:
            try:
                total = int(data.get("total", 0))
            except (TypeError, ValueError):
                total = 0
            log.info(f"[{page_slug}] total reported by API: {total}")

        if not items:
            log.info(f"[{page_slug}] no more items, stopping.")
            break

        new_count = 0
        for item in items:
            sku = item.get("sku")
            if not sku or sku in seen_skus:
                continue
            seen_skus.add(sku)
            collected.append(flatten_product(item, page_slug))
            new_count += 1

        log.info(
            f"[{page_slug}] +{new_count} new products "
            f"(total collected so far: {len(collected)})"
        )

        fetched_so_far = index * PAGE_LIMIT
        if fetched_so_far >= total or new_count == 0:
            break

        index += 1
        polite_sleep()


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    seen_skus = set()
    collected = []

    for slug in CATEGORY_SLUGS:
        crawl_category(slug, seen_skus, collected)

    log.info(f"Finished crawling {len(CATEGORY_SLUGS)} subcategories.")

    if not collected:
        log.error("No products collected. Check API_URL / CATEGORY_SLUGS / field names.")
        return

    df = pd.DataFrame(collected)

    cols = [
        "ProductID", "ProductName", "Category", "CategoryCode", "SourceSlug",
        "Brand", "Price", "OriginalPrice", "DiscountPercent", "Unit",
        "Rating", "TotalSold", "URL",
    ]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    df.to_json(OUTPUT_JSON, orient="records", force_ascii=False, indent=2)

    log.info(
        f"DONE. Collected {len(df)} unique products across "
        f"{len(CATEGORY_SLUGS)} subcategories."
    )
    log.info(f"Non-null counts per column:\n{df.count()}")

    if len(df) < TARGET_RECORDS:
        log.warning(
            f"Only got {len(df)} records, target was {TARGET_RECORDS}. "
            f"Double-check CATEGORY_SLUGS matches the real subcategory list."
        )


if __name__ == "__main__":
    main()
