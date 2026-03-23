#!/usr/bin/env python3
"""
Tạo license key cho khách hàng.
CHỈ BẠN GIỮ FILE NÀY — KHÔNG GỬI CHO KHÁCH.

Cách dùng:
    python generate_key.py                    # key 3 ngày, không tên
    python generate_key.py 7                  # key 7 ngày
    python generate_key.py 30 "Truong THPT A" # key 30 ngày, có tên khách
"""

import sys
from license_manager import generate_key, verify_key
from datetime import datetime

def main():
    days = 3
    customer = ""

    if len(sys.argv) >= 2:
        try:
            days = int(sys.argv[1])
        except ValueError:
            print(f"[LỖI] Số ngày phải là số nguyên, nhận được: {sys.argv[1]}")
            sys.exit(1)

    if len(sys.argv) >= 3:
        customer = sys.argv[2]

    key = generate_key(days=days, customer_name=customer)
    result = verify_key(key)

    print("=" * 60)
    print("  TẠO LICENSE KEY THÀNH CÔNG")
    print("=" * 60)
    print(f"  Khách hàng : {customer or '(không tên)'}")
    print(f"  Thời hạn   : {days} ngày")
    print(f"  Hết hạn    : {result.get('expire_date', 'N/A')}")
    print(f"  Tạo lúc    : {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print("=" * 60)
    print()
    print(f"  KEY: {key}")
    print()
    print("=" * 60)
    print("  Gửi KEY trên cho khách. KHÔNG gửi file generate_key.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
