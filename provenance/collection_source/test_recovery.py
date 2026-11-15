import unittest,tempfile,json,hashlib
from pathlib import Path
from types import SimpleNamespace as S
from qwen_recovery import provider_class
class Usage:
 def model_dump(self,**kw):return {'cost':.01}
class Fake:
 responses=[]
 def __init__(self):self.client=S(max_retries=2);self.caps=[]
 def _request_with_retries(self,request):
  self.caps.append(request['max_tokens']);text,finish=self.responses.pop(0)
  return S(choices=[S(message=S(content=text),finish_reason=finish)],usage=Usage(),model='qwen/qwen3.6-plus')
class Tests(unittest.TestCase):
 def check_case(self,seq,expected,base=8192,prior=False):
  with tempfile.TemporaryDirectory()as d:
   if prior:
    row={'prompt_sha256':hashlib.sha256(b'p').hexdigest(),'image_sha256':[],'response_text':'','provider_metadata':{'finish_reason':'length','request_parameters':{'max_tokens':base},'usage':{'cost':.01}}}
    (Path(d)/'provider_calls.jsonl').write_text(json.dumps(row)+'\n')
   cls=provider_class(Fake,d);o=cls();o.responses=list(seq);out=o._request_with_retries({'max_tokens':base,'messages':[{'role':'user','content':[{'type':'text','text':'p'}]}]})
   self.assertEqual(o.caps,expected);self.assertAlmostEqual(o._logical_cost,.01*len(expected));return out
 def test_nonempty_incorrect_retained(self):self.assertEqual(self.check_case([('invalid answer','length')],[8192]).choices[0].message.content,'invalid answer')
 def test_only_empty_length_recovers(self):self.check_case([('','stop')],[8192])
 def test_bounded_expansion(self):self.check_case([('','length'),('','length'),('','length')],[8192,16384,32768])
 def test_trace_cap(self):self.check_case([('','length'),('answer','stop')],[16384,32768],16384)
 def test_resume_saved_empty(self):self.check_case([('','length'),('answer','stop')],[16384,32768],prior=True)
if __name__=='__main__':unittest.main()
