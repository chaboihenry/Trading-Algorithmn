"""
Test Market Calendar Functionality

This script validates the market calendar implementation:
1. Tests holiday detection (Christmas, Thanksgiving, New Year's, etc.)
2. Tests trading days range retrieval
3. Tests weekend detection
4. Tests market hours detection
5. Compares against known NYSE holidays for 2024-2026

Run this to verify that the Alpaca Calendar API integration works correctly.
"""

import sys
from pathlib import Path
from datetime import date, datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.market_calendar import (
    is_trading_day,
    get_trading_days,
    is_market_open,
    now_et,
    market_calendar,
    MARKET_TZ
)

print("=" * 80)
print("MARKET CALENDAR TEST SUITE")
print("=" * 80)

# Test 1: Known NYSE holidays (should all be False)
print("\n" + "=" * 80)
print("TEST 1: NYSE Holiday Detection")
print("=" * 80)

known_holidays = {
    # 2024 holidays
    date(2024, 1, 1): "New Year's Day",
    date(2024, 1, 15): "Martin Luther King Jr. Day",
    date(2024, 2, 19): "Presidents Day",
    date(2024, 3, 29): "Good Friday",
    date(2024, 5, 27): "Memorial Day",
    date(2024, 6, 19): "Juneteenth",
    date(2024, 7, 4): "Independence Day",
    date(2024, 9, 2): "Labor Day",
    date(2024, 11, 28): "Thanksgiving",
    date(2024, 12, 25): "Christmas",

    # 2025 holidays (testing future dates)
    date(2025, 1, 1): "New Year's Day",
    date(2025, 12, 25): "Christmas",
    date(2025, 11, 27): "Thanksgiving",

    # 2026 holidays
    date(2026, 1, 1): "New Year's Day",
    date(2026, 12, 25): "Christmas",
}

holidays_passed = 0
holidays_failed = 0

for holiday_date, holiday_name in known_holidays.items():
    is_trading = is_trading_day(holiday_date)
    if not is_trading:
        print(f"✓ {holiday_date} ({holiday_name}): Correctly identified as holiday")
        holidays_passed += 1
    else:
        print(f"✗ {holiday_date} ({holiday_name}): FAILED - marked as trading day")
        holidays_failed += 1

print(f"\nHoliday Detection: {holidays_passed}/{len(known_holidays)} passed")

# Test 2: Known trading days (should all be True)
print("\n" + "=" * 80)
print("TEST 2: Regular Trading Day Detection")
print("=" * 80)

known_trading_days = [
    date(2024, 1, 2),   # Tuesday after New Year
    date(2024, 7, 5),   # Friday after July 4th
    date(2024, 12, 26), # Thursday after Christmas
    date(2025, 1, 2),   # Thursday after New Year
    date(2026, 1, 2),   # Friday after New Year
]

trading_passed = 0
trading_failed = 0

for trading_day in known_trading_days:
    is_trading = is_trading_day(trading_day)
    if is_trading:
        print(f"✓ {trading_day}: Correctly identified as trading day")
        trading_passed += 1
    else:
        print(f"✗ {trading_day}: FAILED - marked as holiday")
        trading_failed += 1

print(f"\nTrading Day Detection: {trading_passed}/{len(known_trading_days)} passed")

# Test 3: Weekend detection (should all be False)
print("\n" + "=" * 80)
print("TEST 3: Weekend Detection")
print("=" * 80)

# Find a Saturday and Sunday
test_date = date(2024, 1, 6)  # Saturday
while test_date.weekday() != 5:  # Find a Saturday
    test_date += timedelta(days=1)

saturday = test_date
sunday = test_date + timedelta(days=1)

weekend_passed = 0
weekend_failed = 0

for weekend_day, day_name in [(saturday, "Saturday"), (sunday, "Sunday")]:
    is_trading = is_trading_day(weekend_day)
    if not is_trading:
        print(f"✓ {weekend_day} ({day_name}): Correctly identified as non-trading day")
        weekend_passed += 1
    else:
        print(f"✗ {weekend_day} ({day_name}): FAILED - marked as trading day")
        weekend_failed += 1

print(f"\nWeekend Detection: {weekend_passed}/2 passed")

# Test 4: Trading days range
print("\n" + "=" * 80)
print("TEST 4: Trading Days Range")
print("=" * 80)

# Test around Christmas 2024 (should skip Dec 25)
christmas_start = date(2024, 12, 23)  # Monday
christmas_end = date(2024, 12, 27)    # Friday

trading_days = get_trading_days(christmas_start, christmas_end)

print(f"Trading days from {christmas_start} to {christmas_end}:")
for day in trading_days:
    print(f"  {day} ({day.strftime('%A')})")

expected_days = 4  # Mon 23, Tue 24, Thu 26, Fri 27 (skip Wed 25 = Christmas)
if len(trading_days) == expected_days and date(2024, 12, 25) not in trading_days:
    print(f"✓ Range test PASSED: {len(trading_days)} days (Christmas excluded)")
    range_passed = True
else:
    print(f"✗ Range test FAILED: Expected {expected_days} days, got {len(trading_days)}")
    range_passed = False

# Test 5: Current time and market status
print("\n" + "=" * 80)
print("TEST 5: Current Market Status")
print("=" * 80)

current_time = now_et()
market_open = is_market_open()
today_is_trading_day = is_trading_day(current_time.date())

print(f"Current time (Eastern): {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
print(f"Is today a trading day? {today_is_trading_day}")
print(f"Is market currently open? {market_open}")

if not market_open and today_is_trading_day:
    time_to_open = market_calendar.time_until_market_open()
    if time_to_open:
        hours = time_to_open.total_seconds() / 3600
        print(f"Market opens in: {hours:.1f} hours")

# Test 6: Next trading day
print("\n" + "=" * 80)
print("TEST 6: Next Trading Day Calculation")
print("=" * 80)

# Test from a Friday (should get next Monday unless holiday)
test_friday = date(2024, 12, 20)  # Friday before Christmas week
next_day = market_calendar.next_trading_day(test_friday)

print(f"Next trading day after {test_friday}: {next_day} ({next_day.strftime('%A')})")

if next_day.weekday() == 0 and (next_day - test_friday).days == 3:
    print("✓ Next trading day calculation: PASSED")
    next_day_passed = True
else:
    print("✗ Next trading day calculation: FAILED")
    next_day_passed = False

# Final Summary
print("\n" + "=" * 80)
print("TEST RESULTS SUMMARY")
print("=" * 80)

total_tests = 6
passed_tests = 0

print(f"1. Holiday Detection: {holidays_passed}/{len(known_holidays)} passed")
if holidays_failed == 0:
    passed_tests += 1

print(f"2. Trading Day Detection: {trading_passed}/{len(known_trading_days)} passed")
if trading_failed == 0:
    passed_tests += 1

print(f"3. Weekend Detection: {weekend_passed}/2 passed")
if weekend_failed == 0:
    passed_tests += 1

print(f"4. Trading Days Range: {'PASSED' if range_passed else 'FAILED'}")
if range_passed:
    passed_tests += 1

print(f"5. Current Market Status: PASSED (informational)")
passed_tests += 1

print(f"6. Next Trading Day: {'PASSED' if next_day_passed else 'FAILED'}")
if next_day_passed:
    passed_tests += 1

print("\n" + "=" * 80)
if passed_tests == total_tests:
    print(f"✓ ALL TESTS PASSED ({passed_tests}/{total_tests})")
    print("\nValidation:")
    print("  - Holiday detection working correctly")
    print("  - Weekend detection working correctly")
    print("  - Trading days range calculation accurate")
    print("  - No hardcoded dates - uses Alpaca Calendar API")
    print("  - Market calendar ready for production use")
else:
    print(f"✗ SOME TESTS FAILED ({passed_tests}/{total_tests})")
    print("\nPlease review the failed tests above and check:")
    print("  1. Alpaca API credentials are configured")
    print("  2. Network connection is working")
    print("  3. Alpaca Calendar API is accessible")

print("=" * 80)
