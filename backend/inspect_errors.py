
import numpy as np

errors_file = np.load("models/orby_validation_errors.npz")

zero_mad_channels = []

print("VALIDATION ERROR DISTRIBUTION")
print("-" * 100)

for channel in sorted(errors_file.files):
    errors = errors_file[channel]

    median = np.median(errors)
    mad = np.median(np.abs(errors - median))

    p95 = np.percentile(errors, 95)
    p99 = np.percentile(errors, 99)
    p995 = np.percentile(errors, 99.5)
    p999 = np.percentile(errors, 99.9)

    if mad < 1e-6:
        zero_mad_channels.append(channel)

    print(
        f"{channel:6} | "
        f"N={len(errors):6} | "
        f"P95={p95:9.4f} | "
        f"P99={p99:9.4f} | "
        f"P99.5={p995:9.4f} | "
        f"P99.9={p999:9.4f} | "
        f"MAX={np.max(errors):9.4f}"
    )

print("\nSUMMARY")
print("Total channels:", len(errors_file.files))
print("Channels with near-zero MAD:", len(zero_mad_channels))
print("Near-zero MAD channels:", zero_mad_channels)
