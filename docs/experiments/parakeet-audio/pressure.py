import datetime
import pathlib
import subprocess

def snapshot():
    memory = dict(line.split(':', 1) for line in pathlib.Path('/proc/meminfo').read_text().splitlines())
    vm = dict(line.split() for line in pathlib.Path('/proc/vmstat').read_text().splitlines())
    gpu = subprocess.check_output([
        'nvidia-smi', '--query-gpu=memory.used,memory.total,memory.free,utilization.gpu,utilization.memory',
        '--format=csv,noheader,nounits'], text=True, timeout=5).strip().split(', ')
    pressure = pathlib.Path('/proc/pressure/memory').read_text()
    full = float(pressure.split('full avg10=')[1].split()[0])
    applications = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
        '--format=csv,noheader,nounits'], text=True, timeout=5)
    process_gpu = {}
    for line in applications.splitlines():
        pid, used = line.split(', ')
        if used.isdigit():
            process_gpu[pid] = int(used)
    return dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                available_ram_kib=int(memory['MemAvailable'].split()[0]),
                swap_used_kib=int(memory['SwapTotal'].split()[0]) - int(memory['SwapFree'].split()[0]),
                swapout_pages=int(vm['pswpout']), memory_pressure_full_avg10=full,
                gpu_used_mib=int(gpu[0]), gpu_total_mib=int(gpu[1]),
                gpu_free_mib=int(gpu[2]), gpu_util_percent=int(gpu[3]), gpu_memory_util_percent=int(gpu[4]),
                process_gpu_mib=process_gpu, loadavg=pathlib.Path('/proc/loadavg').read_text().strip(),
                cpu_ticks=list(map(int, pathlib.Path('/proc/stat').read_text().splitlines()[0].split()[1:])))


def pressure_reason(state, previous):
    if state['available_ram_kib'] < 1024**2:
        return 'Available system RAM below 1 GiB'
    if state['gpu_free_mib'] < 1024:
        return 'Total GPU headroom below 1 GiB'
    if state['memory_pressure_full_avg10'] > 1:
        return 'System memory full-stall pressure above 1%'
    if state['swapout_pages'] - previous['swapout_pages'] > 32768:
        return 'System swapped out more than 128 MiB between polls'
    return None
