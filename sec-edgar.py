"""Copyright (C) 2015 Chukwuchebem Orakwue

Licensed under the Apache License, Version 2.0 (the 'License');
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an 'AS IS' BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

try:
    from cStringIO import StringIO
except:
    from StringIO import StringIO
import ftplib
import tempfile
import logbook
import os
import re
import urllib2
import json
from pandas import read_csv, DataFrame, period_range, Timestamp
from pandas.tseries.offsets import QuarterEnd

logbook.set_datetime_format('local')
logger = logbook.Logger('SEC-EDGAR')

try:
    from xantos import INDICES
    index = INDICES.DOW
except ImportError:
    logger.warn('Unable to import xantos')
    index = None

class SECEdgar(object):
    """
    Fetches filings typically 8-K forms from SEC Edgar database.
    Used to create 8-k corpus for (NLP) sentiment analysis.

    Use at your own risk. SEC anonymous FTP prefers queries during market hours.
    """
    FTP_ADDR = "ftp.sec.gov"
    EDGAR_PREFIX = "ftp://%s/" % FTP_ADDR
    COLUMNS = 'CIK|Company Name|Form Type|Date Filed|Filename'.split('|')

    CACHE_FORMAT = 'SEC-EDGAR-FILLING-INDEX-{start}-{end}.csv'
    IDX_FORMAT = 'edgar/full-index/{year}/QTR{qtr}/master.idx'

    ER_CORPUS_DIR = os.path.join(os.path.expanduser("~"),
        'sec_edgar',
        'data'
    )

    def __init__(self, start_year=1993):
        self.start_year = start_year
        self.ftp = ftplib.FTP(self.FTP_ADDR)

        self._last_quarter = Timestamp('now') - QuarterEnd(1)
        self.periods = period_range(self.start_year,
                                    self._last_quarter,
                                    freq='Q')

        self._start = '{year}Q{qtr}'.format(year=self.periods[0].year,
                                            qtr=self.periods[0].quarter)

        self._end = '{year}Q{qtr}'.format(year=self.periods[-1].year,
                                          qtr=self.periods[-1].quarter)
        self.CACHE = os.path.join(self.ER_CORPUS_DIR,
                                  self.CACHE_FORMAT.format(start=self._start,
                                                           end=self._end))

        self.archive = DataFrame(columns=self.COLUMNS)


    def run(self, form_type='8-K', index=None, ensure_func=None):
        SECEdgar.mkdir(self.ER_CORPUS_DIR)
        self.ftp.login()
        self.get_listings()
        self.get_fillings(form_type=form_type, index=index, ensure_func=ensure_func)
        self.ftp.close()

    @property
    def idx_listings(self):
        """
        Returns generator that is list of quarterly idx files archived in EDGAR.
        This is Master Index of EDGAR Dissemination Feed.
        """
        return (self.IDX_FORMAT.format(year=period.year, qtr=period.quarter) \
                for period in self.periods)

    def ftp_retr(self, filename, buffer):
        """Write remote filename's bytes from ftp to local buffer """
        self.ftp.retrbinary('RETR %s' % filename, buffer.write)
        return buffer

    def fix_archive(self):
        self.archive['Company Name'] = \
            self.archive['Company Name'].apply(SECEdgar.clean_company_name)

        self.archive = self.archive.set_index('CIK')
        self.archive.drop('Unnamed: 0', inplace=True)

    def get_listings(self):
        """
        Download all idx files at once
        """

        if os.path.exists(self.CACHE):
            logger.info("Fetching from cache.")
            self.archive = read_csv(self.CACHE)
            self.fix_archive()
        else:
            logger.info("Index download started.")

            for _file in self.idx_listings:
                self.archive = self.archive.append(self.text_to_dataframe(self.get(_file)))

            #self.ftp.close()
            logger.info("Index download complete.")
            self.fix_archive()
            self.archive.to_csv(self.CACHE)
        return self.archive

    def get(self, file, skip_headers=True):
        """
        Download and returns content of an idx file from SEC EDGAR Database.
        """
        logger.info('Downloading {}'.format(file))

        with tempfile.TemporaryFile() as tmp:
            self.ftp_retr(file, tmp)
            tmp.seek(0)
            if skip_headers:
                for x in xrange(0,10):
                    tmp.readline()
                tmp.seek(0, 1)
            return tmp.read()

    def text_to_dataframe(self, text):
        df = read_csv(StringIO(text), sep='|', header=None)
        df.columns = self.COLUMNS
        return df

    @staticmethod
    def mkdir(dir_path):
        if not os.path.isdir(dir_path):
            try:
                os.makedirs(dir_path)
            except:
                pass

    @staticmethod
    def clean_company_name(company_name):
        return re.sub('/\w+/', '', company_name).strip().upper()

    def get_fillings(self, form_type='8-K', index=None, ensure_func=None):
        """Fetches form-8-Ks

        index: Index, Default <None>
            If provided, downloads filings for companys in the index/
        ensure_func: function, Default <None>
            Function to return on filing to ensure its what's expected.

        """

        sec_archive = self.archive[self.archive['Form Type'] == form_type]

        ciks = {}
        ciks_to_symbol = {}
        if index:
            # Expects index object from xantos plaform.
            index.initialize()
            index_components = index.components.index.tolist()

            for symbol in index_components:
                cik = SECEdgar.get_cik(symbol)
                if cik:
                    ciks[symbol] = cik
                    ciks_to_symbol[cik] = symbol

        # If error, then I'd be damned, someone didnt follow SEC rules
        sec_archive = sec_archive.ix[ciks.values()] if ciks else sec_archive

        for comp, group in sec_archive.groupby('Company Name'):
            cik = int(group.index.unique()[0])
            if cik in ciks_to_symbol:
                symbol = ciks_to_symbol[cik]
            else:
                symbol = SECEdgar.get_ticker(comp)
            # ignore non-publicly traded entities
            if not symbol:
                continue

            logger.info('Getting Form {form_type} for {symbol}'.format(symbol=symbol,
                        form_type=form_type))

            ticker_dir = os.path.join(self.ER_CORPUS_DIR, symbol)
            self.mkdir(ticker_dir)

            for idx, _file in enumerate(group['Filename']):

                filepath = os.path.join(ticker_dir,
                    '{symbol}-{filingdate}-{form_type}.txt'.format(
                        symbol=symbol,
                        filingdate=group['Date Filed'].iloc[idx],
                        form_type=form_type))

                # Already exists, skip it
                if os.path.exists(filepath):
                    continue

                contents = self.get(_file, skip_headers=False)

                # Don't save if content is ehn?
                if ensure_func and not ensure_func(contents):
                    continue

                with open(filepath, 'w') as form:
                    form.write(contents)

    @staticmethod
    def get_cik(symbol):
      """Returns Central Index Key (CIK) used in SEC EDGAR to identify
      companies and individuals. None otherwise"""
      URL = "http://finance.yahoo.com/q/sec?s=%s+SEC+Filings" % (symbol)
      try:
          return int(re.findall('=[0-9]*',
                                str(re.findall('cik=[0-9]*',
                                               urllib2.urlopen(URL).read())[0]))[0][1:])
      except:
          logger.warn('Unable to get CIK for {}'.format(symbol))
          return

    @staticmethod
    def get_ticker(company_name):
        """Given company name - return ticker(s).
        Indirect way to check if company if publicly traded.
        """
        from urllib import quote

        logger.info('Getting symbol for {}'.format(company_name))

        def _decode_list(data):
            rv = []
            for item in data:
                if isinstance(item, unicode):
                    item = item.encode('utf-8')
                elif isinstance(item, list):
                    item = _decode_list(item)
                elif isinstance(item, dict):
                    item = _decode_dict(item)
                rv.append(item)
            return rv

        def _decode_dict(data):
            rv = {}
            for key, value in data.iteritems():
                if isinstance(key, unicode):
                    key = key.encode('utf-8')
                if isinstance(value, unicode):
                    value = value.encode('utf-8')
                elif isinstance(value, list):
                    value = _decode_list(value)
                elif isinstance(value, dict):
                    value = _decode_dict(value)
                rv[key] = value
            return rv

        url = 'http://d.yimg.com/autoc.finance.yahoo.com/autoc?query=%s&callback=YAHOO.Finance.SymbolSuggest.ssCallback'% quote(company_name)
        try:
            query_result = urllib2.urlopen(url).read()
            query_result = query_result[len('YAHOO.Finance.SymbolSuggest.ssCallback('):-1]
            raw_dict = json.loads(query_result, object_hook=_decode_dict)['ResultSet']['Result']
            # Fuzzy logic to only return equity
            for res in raw_dict:
                if res['typeDisp'] == 'Equity':
                    return res['symbol']
        except (ValueError, urllib2.HTTPError):
            pass

def ensure_er(text):
    """Ensure text is earnings release i.e. has `Exhibit 99<.1>`"""
    pattern = re.compile('.*Ex.*99.*', re.IGNORECASE)
    return re.search(pattern, text)

def main():

    # Grab SEC archive for 8-K Forms for all companies in provided index
    sec_edgar = SECEdgar(start_year=2000)
    sec_edgar.run(index=index, ensure_func=ensure_er)

if __name__ == '__main__':
    main()
