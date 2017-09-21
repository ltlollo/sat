#include <iostream>
#include <fstream>
#include <sstream>
#include <assert.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <iterator>
#include <stdio.h>

using Can = uint8_t;
using Term = std::vector<Can>;
using CubeList = std::vector<Term>;
using Vec2 = std::vector<std::vector<int>>;

struct Data {
	int nvars;
	Vec2 ele;
};

void
print_term(const Term &t, const char *sepa = " ") {
	putchar('[');
	for (int i = 0; ; i++) {
		printf("%d%d", t[i] >> 1, t[i] & 1);
		if (i == t.size() - 1) {
			break;
		}
		printf("%s", sepa);
	}
	putchar(']');
}

void
print_cube_list(const CubeList &cl, const char *sepa = ", ") {
	putchar('[');
	for (int i = 0; ; i++) {
		print_term(cl[i]);
		if (i == cl.size() - 1) {
			break;
		}
		printf("%s", sepa);
	}
	putchar(']');
}

Term
one(int size) {
	return std::vector<Can>(size, 0b11);
}

bool
is_one(const CubeList &cl) {
	return std::any_of(cl.begin(), cl.end(), [](const Term &t) {
		return std::all_of(t.begin(), t.end(), [](const Can &v) {
				return v == 0b11;
		});
	});
}

Can
neg(Can v) {
	return (~v) & 0b11;
}

int
binate_pivot(int nvars, const CubeList &cl) {
	std::vector<int> pv(nvars);
	std::vector<int> nv(nvars);

	for (const auto &t : cl) {
		for (int i = 0; i < nvars; i++) {
			bool f = t[i] & 1, s = (t[i] >> 1) & 1;
			
			pv[i] += f && !s;
			nv[i] += s && !f;
		}
	}
	
	// optimiazble just compute sum, diff
	int max = pv[0] + nv[0];
	int min = std::abs(pv[0] - nv[0]);
	int candidate = 0;

	for (int i = 1; i < nvars; i++) {
		int sum = pv[i] + nv[i];
		int delta = std::abs(pv[i] - nv[i]);
		if (max < sum) {
			max = sum;
			min = delta;
			candidate = i;
		} else if (max == sum) {
			if (delta < min) {
				max = sum;
				min = delta;
				candidate = i;
			}
		}
	}
	return candidate;
}

Term
term_pos(int nvars, int pos, int val) {
	auto res = one(nvars);
	res[pos] = val;
	return res;
}

void
cofactor_pos(CubeList &cl, int pos, int val) {
	auto n = neg(val);

	cl.erase(std::remove_if(cl.begin(), cl.end(), [pos, n](const Term &t) {
		return t[pos] == n;
	}), cl.end());
	for (auto &t : cl) {
		t[pos] |= n;
	}
}

void
and_pos(CubeList &cl, int pos, int val) {
	if (cl.empty()) {
		return;
	}
	cl.erase(std::remove_if(cl.begin(), cl.end(), [pos, val](Term &t) {
		t[pos] &= val;
		return t[pos] == 0;
	}), cl.end());
}

void
or_inplace_move(CubeList &fst, CubeList &snd) {
	fst.insert(fst.end(), std::make_move_iterator(snd.begin()),
			   std::make_move_iterator(snd.end()));
}


void
complement(int nvars, CubeList &cl) {
	if (cl.empty()) {
		cl.emplace_back(one(nvars));
		return;
	}
	if (is_one(cl)) {
		cl.clear();
		return;
	}
	if (cl.size() == 1) {
		Term t = cl[0];
		cl.clear();
		for (int i = 0; i < t.size(); i++) {
			if (t[i] != 0b11) {
				cl.emplace_back(term_pos(nvars, i, neg(t[i])));
			}
		}
		return;
	}
	int bnp = binate_pivot(nvars, cl);
	
	CubeList cl_copy = cl;
#pragma omp parallel sections
	{
#pragma omp section
		{
			cofactor_pos(cl, bnp, 0b01);
			complement(nvars, cl);
			and_pos(cl, bnp, 0b01);
		}
#pragma omp section
		{
			cofactor_pos(cl_copy, bnp, 0b10);
			complement(nvars, cl_copy);
			and_pos(cl_copy, bnp, 0b10);
		}
	}
	or_inplace_move(cl, cl_copy);
}

void
print_var(const char *v, int num, int neg) {
	if (num == 0) {
		printf("%s\xe2\x82\x80%s", v, neg ? "\xe2\x80\xb2" : "");
		return;
	}
	printf("%s", v);
	for (int o = 1000000000; o; o /= 10) {
		if (num < o) continue;
		int val = num / o;
		printf("\xe2\x82%c", '\x80' + val);
		num -= val * o;
	}
	printf("%s",  neg ? "\xe2\x80\xb2" : "");
}

void
print_input_data(int nvars, const Vec2 &data) {
	if (data.empty()) {
		printf("1\n");
	}
	for (size_t v = 0; v < data.size() - 1; v++) {
		for (size_t i = 0; i < data[v].size(); i++) {
			int vsign = data[v][i] > 0;
			int vnum = std::abs(data[v][i]) - 1;
			assert(vnum < nvars);
			print_var("x", vnum, !vsign);
			printf(" ");
		}
		printf("+\n");
	}
	int v = data.size() - 1;
	for (int i = 0; i < data[v].size(); i++) {
		int vsign = data[v][i] > 0;
		int vnum = std::abs(data[v][i]) - 1;
		assert(vnum < nvars);
		print_var("x", vnum, !vsign);
		printf(" ");
	}
	printf("\n");
}

CubeList
cube_list_fromvec(int nvars, const Vec2 &data) {
	CubeList res(data.size(), one(nvars));

	for (int v = 0; v < data.size(); v++) {
		for (int i = 0; i < data[v].size(); i++) {
			int pos = std::abs(data[v][i]) - 1;
			int sign = data[v][i] < 0;
			res[v][pos] &= (1 << sign);
		}
	}

	return res;
}

CubeList
cube_list_fromdata(const Data &data) {
	return cube_list_fromvec(data.nvars, data.ele);
}

void
print_foramtted_output(const char *fname, int nvars, const CubeList &cl) {
	FILE *fdst = fopen(fname, "w");
	assert(fdst);

	fprintf(fdst, "%d\n", nvars);
	fprintf(fdst, "%d\n", (int)cl.size());

	for (const auto &t : cl) {
		int dont_care = 0;
		for (int i = 0; i < nvars; i++) {
			if (t[i] == 0b11) dont_care++;
		}
		fprintf(fdst, "%d ", nvars - dont_care);
		for (int i = 0; i < nvars; i++) {
			if (t[i] == 0b01) {
				fprintf(fdst, "%d ", i+1);
			} else if (t[i] == 0b10) {
				fprintf(fdst, "%d ", -i-1);
			}
		}
		fprintf(fdst, "\n");
	}
	fprintf(fdst, "\n");
}

Data
parse_function_file(const char *fname) {
	Data res;
	Vec2 &data = res.ele;
	int &nvars = res.nvars;
	std::istringstream iss;

	assert(fname);
	std::ifstream f(fname);
	std::string s;

	auto& l0 = std::getline(f, s);
	assert(l0);

	iss = std::istringstream(s);
	iss >> nvars;
	assert(nvars);

	int nterms;
	auto &l1 = std::getline(f, s);
	assert(l1);

	iss = std::istringstream(s);
	iss >> nterms;

	if (nterms == 0) {
		return res;
	}
	std::vector<int> v;
	int nele;

	while (std::getline(f, s)) {
		if (s.empty()) {
			break;
		}
		v.clear();
		iss = std::istringstream(s);
		iss >> nele;
		v.resize(nele);

		int ele;
		for (int i = 0; i < nele; i++) {
			iss >> ele;
			v[i] = ele;
		}
		data.push_back(v);
	}
	assert(nterms == data.size());

	return res;
}


void
exec_cmdfile(const char *fname) {
	std::ifstream f(fname);
	std::istringstream iss;
	std::string s;
	std::string function_fname;
	std::unordered_map<int, CubeList> funs;
	Data data;

	char cmd;
	int dst;
	int src[2];

	while (std::getline(f, s)) {
		if (s == "q") {
			break;
		}
		iss = std::istringstream(s);
		iss >> cmd;
		iss >> dst;
		iss >> src[0];
		iss >> src[1];
		
		switch (cmd) {
		case 'r':
			function_fname = std::to_string(dst) + ".pcn";
			data = parse_function_file(function_fname.c_str());
			funs[dst] = cube_list_fromdata(data);
			break;
		case 'p':
			{
				function_fname = std::to_string(dst) + ".pcn";
				auto &in = funs[dst];
				print_foramtted_output(function_fname.c_str(), in[0].size(),
									   in);
			}
			break;
		case '!':
			{
				CubeList in = funs.at(src[0]);
				complement(in[0].size(), in);
				funs[dst] = std::move(in);
			}
			break;
		case '+':
			{
				CubeList in0 = funs.at(src[0]);
				CubeList in1 = funs.at(src[1]);
				or_inplace_move(in0, in1);
				funs[dst] = std::move(in0);
			}
			break;
		case '&':
			{
				CubeList in0;
				CubeList in1;
#pragma omp parallel sections
				{
#pragma omp section
					{
						in0 = funs.at(src[0]);
						complement(in0[0].size(), in0);
					}
#pragma omp section
					{
						in1 = funs.at(src[1]);
						complement(in1[0].size(), in1);
					}
				}
				or_inplace_move(in0, in1);
				complement(in0[0].size(), in0);
				funs[dst] = std::move(in0);
			}
			break;
		}
	}
}

int
main(int argc, char *argv[]) {
	if (argc - 1 != 1) {
		return 1;
	}
	exec_cmdfile(argv[1]);

	return 0;
}
