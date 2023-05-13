#Import libraries
import sqlalchemy as db
import pandas as pd


#Class to work with the database
class DataBase:
    def __init__(self, db_type, user, password, host, db_name):
        self.db_type = db_type
        self.user = user
        self.password = password
        self.host = host
        self.db_name = db_name

        #Create a engine to connect to the database
        self.engine = db.create_engine(f"{db_type}://{user}:{password}@{host}/{db_name}")

        #Connect to the database
        self.connection = self.engine.connect()

    #Function to obtain a table from the database
    def get_table(self, table_name):
        df = pd.read_sql_table(table_name, self.connection)
        return df


    