from klapeyron_py_utils.dataset.csv import CSV


class CSV_c(CSV):
    def __init__(self):
        super(CSV_c, self).__init__()

    csv_col_bb = 'bb'
    CSV_COLUMNS = CSV.CSV_COLUMNS_FILE+[csv_col_bb]
    SAMPLE_COLUMNS_DICT = {
        CSV.SAMPLE_FILE: CSV_COLUMNS
    }

    @staticmethod
    def append_row(csv, path, label, fold, bb):
        csv = super(CSV_c, CSV_c).append_row(csv, path, label, fold, **{CSV_c.csv_col_bb: bb})
        return csv
