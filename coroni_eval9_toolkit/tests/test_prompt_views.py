import importlib.util
import pathlib
import unittest

P=pathlib.Path(__file__).resolve().parents[1]/'make_prompt_view.py'
spec=importlib.util.spec_from_file_location('make_prompt_view',str(P))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class PromptViewTests(unittest.TestCase):
    def test_aes_actions_preserved(self):
        src = '/*license*/\n'+'const int Sbox[10]={'+(','.join(['45']*800))+'};\n'
        src+='void helper(){int x=1; /* unrelated helper */}\n'
        src+='int chstone_aes(){helper();return 0;}\n'
        for i in range(1,7):
            src+='void action%d(){ /*L%d:*/ for(int i=0;i<4;i++){} }\n'%(i,i)
        view,removed,arrays=m.compact(src,'chstone-aes')
        self.assertEqual(set(m.ACTION.findall(view)),set('L%d'%i for i in range(1,7)))
        self.assertIn('helper',removed)
        self.assertIn('extern const int Sbox[10]',view)
        self.assertLess(len(view),len(src))

if __name__=='__main__':unittest.main()
