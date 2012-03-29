import pkgutil
import inphosemantics

package=inphosemantics

for importer, modname, ispkg in pkgutil.walk_packages(
    path=package.__path__, 
    prefix=package.__name__+'.', 
    onerror=lambda x: None):
    print(modname)
