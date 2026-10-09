import io,os,tempfile,unittest,re
os.environ['DATABASE_PATH']=tempfile.mktemp(suffix='.db')
from app import app
class AppTests(unittest.TestCase):
    def setUp(self):
        self.c=app.test_client(); page=self.c.get('/').text
        self.token=re.search('name="csrf-token" content="([^"]+)',page)[1]; self.headers={'X-CSRF-Token':self.token}
    def test_health_and_model(self):
        self.assertEqual(self.c.get('/health').status_code,200)
        for label in ['clean','scratch','spot']:
            image=self.c.get('/sample/'+label).data
            r=self.c.post('/api/inspect',data={'image':(io.BytesIO(image),'sample.png')},headers=self.headers)
            self.assertEqual(r.status_code,200); self.assertEqual(r.json['label'],label); self.assertAlmostEqual(sum(r.json['probabilities'].values()),1,places=3)
    def test_validation(self):
        self.assertEqual(self.c.post('/api/ask',json={'question':'hello'}).status_code,403)
        self.assertEqual(self.c.post('/api/inspect',data={'image':(io.BytesIO(b'invalid'),'x.png')},headers=self.headers).status_code,400)
        self.assertEqual(self.c.post('/api/ask',json={'question':42},headers=self.headers).status_code,400)
    def test_retrieval_isolation(self):
        text=b'Zebra protocol requires quarantining batch QX77 in the purple cabinet before review.'
        r=self.c.post('/api/documents',data={'document':(io.BytesIO(text),'protocol.txt')},headers=self.headers); self.assertEqual(r.status_code,200)
        r=self.c.post('/api/ask',json={'question':'What does zebra protocol require for batch QX77?'},headers=self.headers)
        self.assertIn('QX77',r.json['answer']); self.assertTrue(r.json['sources'])
        other=app.test_client(); html=other.get('/').text; token=re.search('name="csrf-token" content="([^"]+)',html)[1]
        r=other.post('/api/ask',json={'question':'What does zebra protocol require for batch QX77?'},headers={'X-CSRF-Token':token}); self.assertNotIn('QX77',r.json['answer'])
        self.c.delete('/api/documents',headers=self.headers)
        r=self.c.post('/api/ask',json={'question':'zebra QX77 protocol'},headers=self.headers); self.assertNotIn('QX77',r.json['answer'])
    def test_abstention(self):
        r=self.c.post('/api/ask',json={'question':'zqxwplm'},headers=self.headers); self.assertEqual(r.json['mode'],'abstained')
if __name__=='__main__': unittest.main()
