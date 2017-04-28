list_test_scripts = ["test/test_segmentation.py"]

for test_script in list_test_scripts:
    print("Running test {}".format(test_script))
    execfile(test_script)