from selenium import webdriver
import os

url = "./map2.html"
save_dir = "../data/pics_raw"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
browser = webdriver.Chrome()
browser.get(url)
browser.maximize_window()
for i in range(1024):
    name = str(i)
    if len(name) < 4:
        name = "0" * (4 - len(name)) + name
    input("Press enter to screenshot...")
    print("screenshot", i)
    browser.save_screenshot(
        os.path.join(save_dir, "pic%s.png" % (name))
    )
browser.close()