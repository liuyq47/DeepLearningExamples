import argparse
import textwrap

def get_single_config(host_prefix, host_index, dns_name, user_name, pem_path):
    content = textwrap.dedent("""Host %s%d
    Hostname %s
    StrictHostKeyChecking no
    port 22
    user %s
    IdentityFile %s
    IdentitiesOnly yes
    ServerAliveInterval 60
""" % (host_prefix, host_index, dns_name, user_name, pem_path))
    return content

parser = argparse.ArgumentParser(description='Generate SSH config.')
parser.add_argument('--host_prefix', dest='host_prefix', required=True)
parser.add_argument('--hosts_file', dest='hosts_file_path', required=True)
parser.add_argument('--user_name', dest='user_name', required=True)
parser.add_argument('--pem_path', dest='pem_path', required=True)
parser.add_argument('--start_index', dest='start_index', type=int, required=True)

args = parser.parse_args()

with open(args.hosts_file_path) as f:
    host_names = f.readlines()
host_names = [x.strip() for x in host_names]

host_index = args.start_index
for host_name in host_names:
    print(get_single_config(args.host_prefix, host_index,
        host_name, args.user_name, args.pem_path))
    host_index += 1

