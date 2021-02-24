# -*- coding: utf-8 -*-

# !pip install selenium

# !apt-get update 
# !apt install chromium-chromedriver
# !cp /usr/lib/cfrom selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
import json
import pandas as pd
import numpy as np
import time
import random
import os

def sort_file():
  # rearrange files
  dir_link = base_path
  dir_lists = list(filter(check_file, os.listdir(dir_link)))
  if len(dir_lists) == 0:
      return ''
  else:
      dir_lists.sort(key=lambda fn: os.path.getmtime(dir_link + os.sep + fn))
      return os.path.join(base_path, dir_lists[-1])
 
def check_file(filename):
  # ignore the downloading files
  if filename.endswith('.crdownload'):
    return False
  # Only get back the file but not folder
  return os.path.isfile(os.path.join(base_path, filename))

def file_exist(folder,filename):
  dir_link = os.path.join(base_path,folder)
  if not os.path.exists(dir_link):
    return False
  dir_lists = os.listdir(dir_link)
  if filename in dir_lists:
    return True
  return False

def web_scrap_csv():
  start = time.time()
  options = Options()
  options.add_argument('--headless')
  options.add_argument("--lang=en-US")
  options.add_argument("accept-language=en-US")
  options.add_argument('--no-sandbox')
  options.add_argument("--disable-notifications")
  options.add_argument("--incognito")
  # options.add_argument('--disable-dev-shm-usage')
  prefs = {"download.default_directory": base_path, "profile.default_content_setting_values.automatic_downloads":1,'intl.accept_languages': 'en,en_US'}
  options.add_experimental_option('prefs',prefs)

  driver = webdriver.Chrome(ChromeDriverManager().install(),options=options)
  driver.get("https://www3.aia-pt.com.hk/mpf/public/fundperf/fundprices.jspa?mt=MT3")


  select_fund = Select(driver.find_element_by_name('clid:export.id:IA==:IA==:0'))
  total_fund = len(select_fund.options)
  for fund_number in range(1,total_fund):
    
    ### Select a specfic fund ###
    select_fund.select_by_index(fund_number)
    fundname = select_fund.options[fund_number].text.strip()
    print(f'For fund number: {fund_number}, {fundname}')
    print('Time elapsed: %.2f minutes' % ((time.time()-start)/60))

    select_year = Select(driver.find_element_by_name('clid:export.fromDateYMD.year:IA==:IA==:0'))
    for year_index,year_opt in enumerate(select_year.options[1:]):
      try:
        year_index += 1
        year = year_opt.text.strip()
        print(f'For year: {year}')
        cur_year = year

        # Check does the file exist
        if file_exist(fundname,f"{fundname}_{cur_year}0101_{cur_year}1231.csv"):
          continue

        # Input from day
        select = Select(driver.find_element_by_name('clid:export.fromDateYMD.day:IA==:IA==:0'))
        select.select_by_index(1)


        select = Select(driver.find_element_by_name('clid:export.fromDateYMD.month:IA==:IA==:0'))
        select.select_by_index(1)


        select = Select(driver.find_element_by_name('clid:export.fromDateYMD.year:IA==:IA==:0'))
        select.select_by_index(year_index)
        cur_year = year

        #Input to day
        select = Select(driver.find_element_by_name('clid:export.toDateYMD.day:IA==:IA==:0'))
        select.select_by_index(31)


        select = Select(driver.find_element_by_name('clid:export.toDateYMD.month:IA==:IA==:0'))
        select.select_by_index(12)


        select = Select(driver.find_element_by_name('clid:export.toDateYMD.year:IA==:IA==:0'))
        select.select_by_index(year_index)

        time.sleep(random.uniform(0.5, 2))
        
        submit_btn = driver.find_elements_by_xpath("//input[@name='submit1']")[1]
        driver.execute_script("arguments[0].click();", submit_btn)

        # Check if the website is still the same
        select_fund = Select(driver.find_element_by_name('clid:export.id:IA==:IA==:0'))

        old_file = ''
        while not old_file:
          old_file = sort_file()
        new_file = os.path.join(base_path,fundname,f"{fundname}_{cur_year}0101_{cur_year}1231.csv")
        os.renames(old_file, new_file)

      except NoSuchElementException as ne:
        print('### Reach the oldest record of the fund ###')
        driver.quit()
        driver = webdriver.Chrome(ChromeDriverManager().install(),options=options)
        driver.get("https://www3.aia-pt.com.hk/mpf/public/fundperf/fundprices.jspa?mt=MT3")
        select_fund = Select(driver.find_element_by_name('clid:export.id:IA==:IA==:0'))
        break
      except:
        random_num_for_file = random.random()
        new_file = os.path.join(base_path, str(f"{fundname}_{cur_year}0101_{cur_year}1231({random_num_for_file}).csv"))
        os.renames(old_file, new_file)

def main():
  # base_path = r"/content/Fund Data"
  current = os.getcwd()
  base_path = f"{current}\AlphaGen\content\AIA"
  if not os.path.exists(base_path):
    os.mkdir(base_path)
  # base_path =  r"c:\Users\kfung\Documents\GitHub\AlphaGen\content\AIA"

  ### Download the fund prices data from AIA webpages ###
  web_scrap_csv()

  ### Get the files from the storage and parse it into required dict ###
  dir_lists = os.listdir(base_path)
  df_all_dict = {}
  for fund in dir_lists:
    if fund.startswith('.'):
      continue
    fund_files_list = sorted(os.listdir(os.path.join(base_path,fund)))
    fund_df = pd.DataFrame()
    for filename in fund_files_list:
      file_path = os.path.join(base_path,fund,filename)
      # print(file_path)
      df = pd.read_csv(file_path,names=['Date', 'Price'],index_col=[0],parse_dates=[0])
      df['Price'] = df['Price'].str.replace("\t","")
      df = df.iloc[5:,0].astype(float, errors ='ignore')
      # df.index = pd.to_datetime(df.index)
      fund_df = pd.concat([fund_df,df])
    df_all_dict[fund] = fund_df.sort_index().to_dict()[0]

  ### Output dict ###
  with open('fund_data.txt', 'w') as file:
     file.write(json.dumps(df_all_dict))
  output_dict = {'Provider':{'MPF Scheme Name':df_all_dict}}
  
  ### Save the result to file ###
  with open(os.path.join(base_path,'fund_data.txt'), 'w') as file:
     file.write(json.dumps(df_all_dict))
      
  return output_dict

if __name__ == "__main__":
    main()

