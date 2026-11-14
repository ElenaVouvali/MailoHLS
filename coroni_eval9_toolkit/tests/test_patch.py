import importlib.util
import pathlib
import unittest

P=pathlib.Path(__file__).resolve().parents[1]/'patch_inference_views.py'
spec=importlib.util.spec_from_file_location('patch_inference_views',str(P))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class TestPatch(unittest.TestCase):
    def test_inserted_function_compiles(self):
        source='def foo(kernel,code,a,domains):\n        prepared = domains.prepare_source_template(kernel, code, a.sources_root)\n        return prepared\n'
        self.assertEqual(source.count(m.ANCHOR),1)
        s=source.replace(m.ANCHOR,m.ANCHOR+m.PATCH)
        compile(s,'<patched>','exec')

if __name__=='__main__':unittest.main()
