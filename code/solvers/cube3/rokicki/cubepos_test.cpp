/*83:*/
#line 2114 "./cubepos.w"

#include <iostream> 
#include <map> 
#include "cubepos.h"
void check(const cubepos&cp1,const cubepos&cp2,const char*msg){
if(cp1==cp2)
return;
for(int i= 0;i<8;i++)
cout<<" "<<(int)(cp1.c[i])<<" "<<(int)(cp2.c[i])<<endl;
for(int i= 0;i<12;i++)
cout<<" "<<(int)(cp1.e[i])<<" "<<(int)(cp2.e[i])<<endl;
cout<<endl<<msg<<endl;
exit(10);
}

/*:83*//*84:*/
#line 2132 "./cubepos.w"

void recur1(const cubepos&cp,int togo,int canonstate,vector<cubepos> &a){
a.push_back(cp);
if(togo--){
cubepos cp2;
long long mask= cubepos::cs_mask(canonstate);
for(int mv= 0;mv<NMOVES;mv++){
if((mask>>mv)&1){
cp2= cp;
cp2.move(mv);
recur1(cp2,togo,cubepos::next_cs(canonstate,mv),a);
}
}
}
}

/*:84*//*85:*/
#line 2151 "./cubepos.w"

#ifdef HALF
unsigned int allpos[]= {1,18,243,3240,43239,574908,7618438,100803036,
1332343288};
unsigned int c48pos[]= {1,2,9,75,934,12077,159131,2101575,27762103,
366611212};
unsigned int c96pos[]= {1,2,8,48,509,6198,80178,1053077,13890036,
183339529};
#endif
#ifdef QUARTER
unsigned int allpos[]= {1,12,114,1068,10011,93840,878880,8221632,
76843595,717789576};
unsigned int c48pos[]= {1,1,5,25,219,1978,18395,171529,1601725,
14956266,139629194,1303138445};
unsigned int c96pos[]= {1,1,5,17,130,1031,9393,86183,802788,
7482382,69833772,651613601};
#endif

/*:85*//*86:*/
#line 2171 "./cubepos.w"

moveseq random_moveseq(int len){
moveseq r;
for(int i= 0;i<len;i++)
r.push_back(random_move());
return r;
}

/*:86*//*87:*/
#line 2181 "./cubepos.w"

const unsigned int MAXELEMENTS= 100000;
map<cubepos,int> world;
vector<cubepos> q;
int main(int argc,char*argv[]){
cubepos cp,cp2,cp3,cp4;
/*88:*/
#line 2205 "./cubepos.w"

if(sizeof(int)!=4)
error("! this code assumes a 4-byte int throughout");
if(sizeof(short)!=2)
error("! this code assumes a two-byte short");
if(sizeof(cubepos)!=20)
error("! size of cubepos is not 20");
for(int i= 0;i<8;i++)
if(cp.c[i]!=identity_cube.c[i])
error("! bad initial cp");
for(int i= 0;i<12;i++)
if(cp.e[i]!=identity_cube.e[i])
error("! bad initial cp");
for(int i= 0;i<16;i++)
if(cubepos::face_map[i][0]%3!=0)
error("! up down not preserved in first 16");

/*:88*/
#line 2187 "./cubepos.w"

/*89:*/
#line 2227 "./cubepos.w"

cout<<"Verifying f/b moves."<<endl;
for(int i= 0;i<NMOVES_EXT;i++){
cp.move(i);
cp.movepc(i);
check(cp,identity_cube,"problem verifying fb of moves");
}
cout<<"Verifying forward move."<<endl;
for(int i= 0;i<FACES;i++){
for(int j= 0;j<4;j++)
cp.move(i*TWISTS);
check(cp,identity_cube,"problem verifying order of basic generators");
}
cout<<"Verifying bw moves."<<endl;
for(int i= 0;i<FACES;i++){
for(int j= 0;j<4;j++)
cp.movepc(i*TWISTS);
check(cp,identity_cube,"problem verifying order of basic generators 2");
}

/*:89*/
#line 2188 "./cubepos.w"

/*90:*/
#line 2254 "./cubepos.w"

cout<<"Random cube inversion"<<endl;
for(int i= 0;i<100;i++){
cp.randomize();
cp.invert_into(cp2);
cp2.invert_into(cp3);
check(cp,cp3,"Inversion failed.");
}
cout<<"Move inversion"<<endl;
for(int i= 0;i<100;i++){
moveseq ms= random_moveseq(10);
moveseq msi= cubepos::invert_sequence(ms);
cp= identity_cube;
cp2= identity_cube;
for(unsigned int k= 0;k<ms.size();k++){
cp.move(ms[k]);
cp2.move(msi[k]);
}
cp.invert_into(cp3);
check(cp2,cp3,"Invert move sequence failed");
}

/*:90*/
#line 2189 "./cubepos.w"

/*91:*/
#line 2281 "./cubepos.w"

cout<<"Multiplication"<<endl;
for(int i= 0;i<100;i++){
moveseq ms= random_moveseq(10),ms2= random_moveseq(10);
cp= identity_cube;
cp2= identity_cube;
cp3= identity_cube;
for(unsigned int k= 0;k<ms.size();k++){
cp.move(ms[k]);
cp3.move(ms[k]);
}
for(unsigned int k= 0;k<ms2.size();k++){
cp2.move(ms2[k]);
cp3.move(ms2[k]);
}
cubepos::mul(cp,cp2,cp4);
check(cp4,cp3,"Bad product");
cp= identity_cube;
cp2= identity_cube;
cp3= identity_cube;
for(unsigned int k= 0;k<ms.size();k++){
cp.movepc(ms[k]);
cp3.movepc(ms[k]);
}
for(unsigned int k= 0;k<ms2.size();k++){
cp2.movepc(ms2[k]);
cp3.movepc(ms2[k]);
}
cubepos::mulpc(cp,cp2,cp4);
check(cp4,cp3,"Bad product");
}

/*:91*/
#line 2190 "./cubepos.w"

/*92:*/
#line 2316 "./cubepos.w"

cout<<"Test parse move"<<endl;
for(int i= 0;i<100;i++){
moveseq ms= random_moveseq(10);
char movebuf[1000];
char*p= movebuf;
for(unsigned int j= 0;j<ms.size();j++)
cubepos::append_move(p,ms[j]);
const char*pp= movebuf;
moveseq ms2= cubepos::parse_moveseq(pp);
if(ms!=ms2)
error("! bad parse");
}

/*:92*/
#line 2191 "./cubepos.w"

/*93:*/
#line 2333 "./cubepos.w"

cout<<"Testing Singmaster"<<endl;
for(int i= 0;i<100;i++){
char singbuf[1000];
cp.randomize();
strcpy(singbuf,cp.Singmaster_string());
const char*err= cp2.parse_Singmaster(singbuf);
if(err)
error(err);
check(cp,cp2,"! mismatch between parse and gen");
}

/*:93*/
#line 2192 "./cubepos.w"

/*94:*/
#line 2349 "./cubepos.w"

cout<<"Testing remap"<<endl;
for(int i= 0;i<100;i++){
moveseq ms;
int m= (int)(M*myrand());
for(int j= 0;j<1;j++)
ms.push_back(random_move());
cp= identity_cube;
cp2= identity_cube;
for(unsigned int j= 0;j<ms.size();j++){
cp.move(ms[j]);
cp2.move(cubepos::move_map[m][ms[j]]);
}
cp.remap_into(m,cp3);
check(cp2,cp3,"Move map issue");
}

/*:94*/
#line 2193 "./cubepos.w"

/*95:*/
#line 2368 "./cubepos.w"

world.clear();
q.clear();
q.push_back(identity_cube);
world[identity_cube]= 0;
unsigned int qg= 0;
int prevd= -1;
duration();
while(qg<q.size()){
int d= world[q[qg]];
if(d!=prevd){
cout<<"At lev "<<d<<" size "<<(q.size()-qg)<<endl;
#if defined(HALF) || defined(QUARTER)
if(allpos[d]!=q.size()-qg)
error("! bad value");
#endif
if(q.size()> MAXELEMENTS)
break;
prevd= d;
}
for(int i= 0;i<NMOVES;i++){
cp= q[qg];
cp.move(i);
if(world.find(cp)==world.end()){
world[cp]= d+1;
q.push_back(cp);
}
}
qg++;
}
cout<<"Took "<<duration()<<endl;

/*:95*/
#line 2194 "./cubepos.w"

/*96:*/
#line 2402 "./cubepos.w"

world.clear();
q.clear();
q.push_back(identity_cube);
world[identity_cube]= 0;
qg= 0;
prevd= -1;
while(qg<q.size()){
int d= world[q[qg]];
if(d!=prevd){
cout<<"At lev "<<d<<" size "<<(q.size()-qg)<<endl;
#if defined(HALF) || defined(QUARTER)
if(c48pos[d]!=q.size()-qg)
error("! bad value");
#endif
if(q.size()> MAXELEMENTS)
break;
prevd= d;
}
for(int i= 0;i<NMOVES;i++){
cp= q[qg];
cp.move(i);
cp.canon_into48(cp2);
if(world.find(cp2)==world.end()){
world[cp2]= d+1;
q.push_back(cp2);
}
}
qg++;
}

/*:96*/
#line 2195 "./cubepos.w"

/*97:*/
#line 2437 "./cubepos.w"

cout<<"Took "<<duration()<<endl;
world.clear();
q.clear();
q.push_back(identity_cube);
world[identity_cube]= 0;
qg= 0;
prevd= -1;
while(qg<q.size()){
int d= world[q[qg]];
if(d!=prevd){
cout<<"At lev "<<d<<" size "<<(q.size()-qg)<<endl;
#if defined(HALF) || defined(QUARTER)
if(c96pos[d]!=q.size()-qg)
error("! bad value");
#endif
if(q.size()> MAXELEMENTS)
break;
prevd= d;
}
for(int i= 0;i<NMOVES;i++){
cp= q[qg];
cp.move(i);
cp.canon_into96(cp2);
if(world.find(cp2)==world.end()){
world[cp2]= d+1;
q.push_back(cp2);
}
cp= q[qg];
cp.movepc(i);
cp.canon_into96(cp2);
if(world.find(cp2)==world.end()){
world[cp2]= d+1;
q.push_back(cp2);
}
}
qg++;
}
cout<<"Took "<<duration()<<endl;

/*:97*/
#line 2196 "./cubepos.w"

/*98:*/
#line 2481 "./cubepos.w"

world.clear();
long long prevcount= 0;
for(int d= 0;;d++){
q.clear();
double t1= walltime();
recur1(identity_cube,d,CANONSEQSTART,q);
double t2= walltime();
sort(q.begin(),q.end());
double t3= walltime();
vector<cubepos> ::iterator nend= unique(q.begin(),q.end());
double t4= walltime();
long long sz= nend-q.begin();
cout<<"Sequences "<<q.size()<<" positions "<<sz<<endl;
cout<<"At lev "<<d<<" size "<<(sz-prevcount)<<endl;
cout<<"Search "<<(t2-t1)<<" sort "<<(t3-t2)<<" uniq "<<
(t4-t3)<<endl;
#if defined(HALF) || defined(QUARTER)
if(allpos[d]!=sz-prevcount)
error("! bad value");
#endif
prevcount= sz;
if(sz> 3000000)
break;
}
cout<<"Took "<<duration()<<endl;/*:98*/
#line 2197 "./cubepos.w"

}


/*:87*/
