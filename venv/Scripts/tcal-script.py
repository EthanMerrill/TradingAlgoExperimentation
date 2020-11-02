#!c:\users\eth22\documents\github\first_zipline_algo\venv\scripts\python.exe
# EASY-INSTALL-ENTRY-SCRIPT: 'trading-calendars==1.11.11','console_scripts','tcal'
__requires__ = 'trading-calendars==1.11.11'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('trading-calendars==1.11.11', 'console_scripts', 'tcal')()
    )
