import platform
import warnings



OS_NAME = platform.system()

if OS_NAME == "Windows":
    from loaders.ge import load_ge
else:
    warnings.simplefilter("default", ImportWarning)
    warnings.warn("You're not running on Windows, I can't import `load_ge`",
                  ImportWarning)


from loaders.philips import load_philips
