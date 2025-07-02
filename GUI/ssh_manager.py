import paramiko
import time
import gc
from scp import SCPClient
from config import FPGA_USER, FPGA_PASSWORD, PROJECT_CONFIG

def connect_ssh(fpga_ip):
    """Establece la conexión SSH con la FPGA en la IP proporcionada."""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh.connect(fpga_ip, username=FPGA_USER, password=FPGA_PASSWORD, timeout=5)
        return ssh
    except Exception as e:
        print(f"Error de conexión con {fpga_ip}: {e}")
        return None

def execute_command(ssh, command, log_message, wait_time=2):
    try:
        log_message(f"Ejecutando: {command}")
        stdin, stdout, stderr = ssh.exec_command(command)

        while not stdout.channel.exit_status_ready():
            time.sleep(1)

        output = stdout.read().decode()
        error = stderr.read().decode()

        if error:
            log_message(f"Error en {command}: {error}")
        else:
            log_message(f"Comando ejecutado correctamente.")

        time.sleep(wait_time)
        return output

    except paramiko.SSHException as e:
        log_message(f"Error de SSH: {e}")
        return None

def fetch_output_files(ssh, fpga_ip, local_path_damage, config_file, log_message, project_id, signal_type):
    project = PROJECT_CONFIG.get(project_id, {})

    if not project:
        log_message(f"Proyecto desconocido: {project_id}")
        return

    try:
        with SCPClient(ssh.get_transport()) as scp:
            remote_output = project["remote_output"]
            remote_config = project["remote_config"]

            log_message(f"Descargando desde: {fpga_ip}:{remote_output} → {local_path_damage}")
            scp.get(remote_output, local_path_damage)
            log_message(f"Archivo de salida descargado en: {local_path_damage}")

            log_message(f"Descargando configuración desde: {fpga_ip}:{remote_config} → {config_file}")
            scp.get(remote_config, config_file)
            log_message(f"Archivo de configuración descargado en: {config_file}")
    except Exception as e:
        log_message(f"Error en la transferencia SCP: {e}")

def close_ssh_connection(ssh):
    if ssh:
        ssh.close()
        print("Conexión SSH cerrada correctamente.")

def send_file_via_ssh(ssh, local_file_path, remote_file_path, log_message):
    try:
        with SCPClient(ssh.get_transport()) as scp:
            log_message(f"Enviando {local_file_path} → {remote_file_path} ...")
            scp.put(local_file_path, remote_file_path)
            log_message(f"Archivo enviado correctamente a {remote_file_path}")
    except Exception as e:
        log_message(f"Error al enviar archivo: {e}")

def download_file_from_ssh(ssh, remote_path, local_path, log_message):
    try:
        with SCPClient(ssh.get_transport()) as scp:
            log_message(f"Descargando {remote_path} → {local_path}")
            scp.get(remote_path, local_path)
            log_message(f"Descarga completada: {local_path}")
    except Exception as e:
        log_message(f"Error en la descarga SCP: {e}")
