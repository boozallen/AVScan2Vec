import re
import datetime
from dateutil.rrule import rrule, MONTHLY

# Global token definitions
CLS = "<CLS>"
EOS = "<EOS>"
SOW = "<SOW>"
EOW = "<EOW>"
PAD = "<PAD>"
MASK = "<MASK>"
ABS = "<ABSTAIN>"
BEN = "<BENIGN>"
UNK = "<UNKNOWN>"
NO_AV = "<NO_AV>"
NO_DATE = "<NO_DATE>"

# Get months/years from Dec 1969 to present
START_DATE = datetime.date(1969,12,1)
END_DATE = datetime.date.today()
DATES = [dt for dt in rrule(MONTHLY, dtstart=START_DATE, until=END_DATE)]
DATES = [NO_DATE] + [datetime.datetime.strftime(dt, "%Y-%m") for dt in DATES]
DATES_T = {dt: i for i, dt in enumerate(DATES)}
NUM_MONTHS = len(DATES)

# Regex for normalizing AV names
AV_NORM = re.compile(r"\W+")
