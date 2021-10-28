class synthpop_config:

    def __init__(self, acsyear=2013):
        self.acsyear = acsyear

    def pums_storage(self):
        if self.acsyear >= 2018:
            storage = "https://storage.googleapis.com/synthpop-public/PUMS2018/pums_2018_acs5/"
        else:
            storage = "https://s3-us-west-1.amazonaws.com/synthpop-data2/"
        return storage

    def __call__(self):
        return self.pums_storage()


class geog_changes_path:
    def __init__(self, acsyear):
        self.acsyear = acsyear

    def geog_change_storage(self):
        storage = "https://storage.googleapis.com/synthpop-public/support_files/"
        return storage

    def __call__(self):
        return self.geog_change_storage()
