# smoke_test.py - simple import + load_data smoke test
import sys
sys.path.insert(0, '.')
import traceback

def main():
    try:
        from predictions import load_data
    except Exception as e:
        print("ERROR importing load_data from predictions:", e)
        traceback.print_exc()
        return 2

    try:
        df = load_data()
        mem_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        print(f"SMOKE OK: rows={len(df)}, memMB={mem_mb:.1f}")
    except Exception as e:
        print("ERROR running load_data():", e)
        traceback.print_exc()
        return 3

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
