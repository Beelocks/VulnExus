# Full vulnerability scan
./vulnexus.py https://example.com --full-scan

# With threat intelligence
./vulnexus.py https://example.com --threat-intel

# Ethical worm propagation (depth 3)
./vulnexus.py https://example.com --ethical-worm --depth 3

# Binary analysis
./vulnexus.py https://example.com --binary /path/to/binary

# Automatic patch generation
./vulnexus.py https://example.com --auto-patch
