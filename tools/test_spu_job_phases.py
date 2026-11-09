#!/usr/bin/env python3
"""Verify optional job-stage clocks and real jobprof aggregation/reset boundaries."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
root = Path(__file__).resolve().parents[1]
cc = os.environ.get('CC') or shutil.which('clang') or shutil.which('cc')
if not cc:
    raise SystemExit('Set CC to an installed C compiler path')
source = (root/'runtime/spu/spu_workload.c').read_text()
def function(name):
    match = re.search(r'^static void '+name+r'\([^;]+?\)\s*\{', source, re.M)
    assert match, name
    start=source.index('{',match.start()); depth=1; end=start+1
    while depth:
        depth += (source[end]=='{')-(source[end]=='}'); end+=1
    return source[match.start():end]
tables=source[source.index('#define SPU_JOB_PROFILE_SLOTS '):source.index('static int spu_job_profile_period_seconds(')]
fixture='''#include <assert.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "spu_job_profile.h"
static double stamp;
double spu_job_profile_now(void){return stamp;}
'''+tables+function('spu_job_profile_record')+'''
int main(void){
 spu_job_phase_times t={.01,.02,.03,.04,.05};
 stamp=1;spu_job_profile_record(0xAB,32,17,1,1,.15,3,&t);
 stamp=4;spu_job_profile_record(0xAB,32,17,1,1,.15,3,&t);
 for(unsigned i=0;i<SPU_JOB_PROFILE_SLOTS;i++){
  assert(s_job_profiles[i].phase_runs==0);
  assert(s_job_profiles[i].phases.lookup==0 && s_job_profiles[i].phases.reset==0);
  assert(s_job_profiles[i].phases.stage==0 && s_job_profiles[i].phases.execute==0);
  assert(s_job_profiles[i].phases.teardown==0 && s_job_profiles[i].total_seconds==0);
 }
 stamp=7;spu_job_profile_record(0xAB,32,17,1,1,.2,3,NULL);
 puts("job profile aggregation: report snapshot and phase reset PASS");return 0;
}
'''
with tempfile.TemporaryDirectory(prefix='spu-jobphase-') as tmp:
    tmp=Path(tmp); src=tmp/'profile.c'; src.write_text(fixture)
    for name,code in [('phases',root/'runtime/spu/tests/test_spu_job_phases.c'),('profile',src)]:
        binary=tmp/(name+('.exe' if os.name=='nt' else ''))
        subprocess.run([cc,'-std=gnu17','-O1','-UNDEBUG','-Wno-deprecated-declarations',
                        '-I'+str(root/'include'),'-I'+str(root/'runtime/spu'),
                        '-I'+str(root/'runtime/platform'),str(code),'-o',str(binary),
                        *([] if os.name=='nt' else ['-pthread','-lm'])],check=True)
        result=subprocess.run([str(binary)],text=True,capture_output=True,timeout=10)
        print(result.stdout.strip()); assert result.returncode==0,result.stderr
        if name=='profile':
            rows=result.stderr.splitlines()
            assert len([r for r in rows if r.startswith('[jobprof]')])==2,rows
            phases=[r for r in rows if r.startswith('[jobphase]')]
            assert len(phases)==1,rows
            for expected in ['image=17','runs=2','lookup=20.000ms','reset=40.000ms',
                             'stage=60.000ms','execute=80.000ms','teardown=100.000ms']:
                assert expected in phases[0],phases
