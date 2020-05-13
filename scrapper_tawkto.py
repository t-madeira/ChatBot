from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import pyautogui

def _esperar(tempo):
    i = 0
    while i < tempo:
        time.sleep(1)
        i += 1

# Lidar com notificações do Chrome
option = Options()
option.add_argument("--disable-infobars")
option.add_argument("start-maximized")
option.add_argument("--disable-extensions")
# 1=aceitar / 2=recusar
option.add_experimental_option("prefs", {
    "profile.default_content_setting_values.notifications": 2
})

# Abrir página
browser = webdriver.Chrome(chrome_options=option, executable_path="C:\\Users\\thith\Downloads\chromedriver_win32\chromedriver.exe")
browser.maximize_window()
browser.get('https://dashboard.tawk.to/login')

email = "email"
senha = "senha"

browser.find_element_by_xpath("/html/body/div[3]/div[2]/div/form/div[2]/input").send_keys(email)
browser.find_element_by_xpath("/html/body/div[3]/div[2]/div/form/div[3]/input").send_keys(senha)
browser.find_element_by_xpath("/html/body/div[3]/div[2]/div/form/div[4]/input").click()

# Gambiarra pra forçar espera
i = 0
while i < 5:
    time.sleep(1)
    i += 1

# Gambiarra pra forçar entrar na página de conversas
browser.get('https://dashboard.tawk.to/#/messaging/57d84ebe11028a70b198b020')
i = 0
while i < 2:
    pyautogui.press('f5')
    i += 1

# Gambiarra pra forçar espera
i = 0
while i < 5:
    if i < 5:
        time.sleep(1)
    i += 1

# Gambiarra pra rolar até o fim da página
data_primeira_conversa = "13/Sep/2016 17:47"
i = 1
flag = True
while flag == True:
    try:
        if browser.find_element_by_xpath("/html/body/div[8]/div[1]/div/div/div[2]/div[2]/div/div/div[2]/table/tbody/tr[" + str(
        i) + "]/td[5]").text != data_primeira_conversa:
            pyautogui.scroll(-3000)
            i += 3
    except:
        break

# Clicar nas caixinhas e extrair o chat
k = 0
flag = True
while flag == True:

    for i in range (1+k*100, 101+k*100):
        print (i, k)
        try:
            browser.find_element_by_xpath\
                ("/html/body/div[8]/div[1]/div/div/div[2]/div[2]/div/div/div[2]/table/tbody/tr["+str(i+k*10)+"]/td[6]/div/label/i").click()
        except:
            flag = False

    browser.find_element_by_xpath("/html/body/div[8]/div[1]/div/div/div[2]/div[2]/div/div/div[3]/div/ul/li[2]/button").click()
    browser.find_element_by_xpath("/html/body/div[12]/div[2]/div/div[2]/form/section/label[2]/input").send_keys(email)
    browser.find_element_by_xpath("/html/body/div[12]/div[2]/div/div[3]/button[2]").click()

    k += 1
